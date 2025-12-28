# =========================================================
# GAS PRICE PREDICTION WITH ATTENTION-LSTM + MEMORY GATE
# =========================================================

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Attention, Concatenate,
    GlobalAveragePooling1D, Multiply, RepeatVector, Lambda
)
from tensorflow.keras.models import Model, load_model
import joblib
import matplotlib.pyplot as plt
import os
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# ---------- REPLACEMENT: LOAD DATA & PREPARE MEMORY FEATURE ----------
def load_data(mongo_uri, db_name, coll_name, window='15min', mem_len=3):
    print("="*60)
    print("LOADING DATA FROM MONGODB & BUILDING MEMORY FEATURE")
    print("="*60)
    client = MongoClient(mongo_uri)
    coll = client[db_name][coll_name]
    
    docs = list(coll.find({}).sort("block_number", 1))
    df = pd.DataFrame(docs)
    if df.empty:
        raise RuntimeError("No data found in collection.")
    
    df["timestamp"] = pd.to_datetime(df["block_time"], utc=True)

    # Convert Wei to Gwei for gas related cols if present
    for col in ["gas_price", "max_fee_per_gas", "max_priority_fee_per_gas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1e9

    # roll-based "memory_congestion" as fallback (existing rolling mean)
    # use bfill() instead of deprecated fillna(method='bfill')
    df["memory_congestion"] = df["gas_price"].rolling(20).mean().bfill()

    # Create time-slot (rounded to nearest window) to capture "same time of day" groups
    # using '15min' avoids the FutureWarning about 'T' alias
    df["time_slot"] = df["timestamp"].dt.floor(window).dt.time

    # Compute mem_same_slot: for each time_slot group compute shifted rolling mean
    # Use transform (keeps the original index alignment) instead of apply
    def shifted_rolling_mean(s):
        # s is a Series for that time_slot (sorted by timestamp)
        return s.shift(1).rolling(mem_len, min_periods=1).mean()

    # groupby + transform with a lambda that reindexes correctly
    # transform expects a function that returns a Series aligned with input group
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["mem_same_slot"] = df.groupby("time_slot")["gas_price"].transform(
        lambda grp: shifted_rolling_mean(grp).values
    )

    # If any mem_same_slot still NaN (e.g., first occurrences), fill with global mean
    global_mean = df["gas_price"].mean()
    df["mem_same_slot"] = df["mem_same_slot"].fillna(global_mean)

    print(f"✓ Loaded {len(df):,} records ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df

# Slight tweak to prepare_data to ensure we handle mem column safely
def prepare_mem_data(df, seq_len=30):
    feature_cols = ["gas_price", "max_fee_per_gas", "max_priority_fee_per_gas", "memory_congestion"]
    # ensure mem_same_slot exists
    if "mem_same_slot" not in df.columns:
        df["mem_same_slot"] = df["gas_price"].mean()
    df_seq = df[feature_cols + ["mem_same_slot"]].fillna(0).reset_index(drop=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_seq)

    X, M, y = [], [], []
    for i in range(len(df_seq) - seq_len):
        X.append(scaled_data[i:i+seq_len, :len(feature_cols)])
        M.append([scaled_data[i+seq_len, -1]])  # mem_same_slot scaled
        y.append(scaled_data[i+seq_len, 0])     # gas_price scaled

    X = np.array(X)
    M = np.array(M)
    y = np.array(y)

    split = int(len(X) * 0.8)
    return X[:split], M[:split], y[:split], X[split:], M[split:], y[split:], scaler


# =========================================================
# 3. BUILD ATTENTION-LSTM + MEMORY GATE MODEL
# =========================================================
def build_alstm_model(input_shape, mem_dim=1, seq_len=None):
    # input_shape: (seq_len, d_features)
    seq_in = Input(shape=input_shape, name="seq_input")             # (seq_len, d_features)
    mem_in = Input(shape=(mem_dim,), name="mem_input")              # (mem_dim,) usually 1

    # Encoder part (first LSTM over sequence)
    x = LSTM(64, return_sequences=True, name="enc_lstm1")(seq_in)
    x = Dropout(0.2)(x)

    # Self-attention over the LSTM outputs
    attn_out = Attention(name="self_attention")([x, x])             # (seq_len, d_model)
    # Optionally concatenate attention outputs with original sequence
    seq_comb = Concatenate(name="seq_concat")([x, attn_out])       # (seq_len, 2*d_model)

    # Now reduce attention outputs to a fixed context vector (c_enc)
    c_enc = GlobalAveragePooling1D(name="context_pool")(attn_out)   # (d_model,)

    # Process memory input via a small MLP
    m_vec = Dense(8, activation="relu", name="mem_dense1")(mem_in)  # (8,)
    m_vec = Dense(8, activation="relu", name="mem_dense2")(m_vec)  # (8,)

    # Gate: learnable scalar (or vector) showing percentage effect of memory
    gate_input = Concatenate(name="gate_concat")([c_enc, m_vec])   # (d_model + 8,)
    g = Dense(8, activation="sigmoid", name="gate_dense")(gate_input)
    # reduce gate to same dim as m_vec or scalar (here elementwise)
    g = Dense(m_vec.shape[-1], activation="sigmoid", name="gate_out")(g)  # (8,)

    # Effective memory contribution
    m_hat = Multiply(name="mem_effect")([g, m_vec])                 # (8,)

    # Broadcast m_hat across time steps to fuse into sequence before second LSTM
    # RepeatVector requires a known seq_len; pass via argument
    if seq_len is None:
        # Fallback: infer from input_shape
        seq_len = input_shape[0]
    m_tiled = RepeatVector(seq_len, name="mem_tiled")(m_hat)        # (seq_len, 8)

    # Concatenate tiled memory with seq_comb (concat along feature axis)
    seq_fused = Concatenate(name="seq_fused")([seq_comb, m_tiled])  # (seq_len, 2*d_model + 8)

    # Second LSTM (decoder-like block)
    x2 = LSTM(32, return_sequences=False, name="dec_lstm")(seq_fused)
    embed = Dense(16, activation='relu', name='embedding')(x2)
    out = Dense(1, name="gas_output")(embed)

    model = Model(inputs=[seq_in, mem_in], outputs=out, name="ALSTM_Memory")
    #model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =========================================================
# 4. TRAIN & SAVE MODEL (adapted for memory input)
# =========================================================
def train_and_save_model(X_train, M_train, y_train, X_val, M_val, y_val, scaler,
                         model_path="gas_alstm_mem.keras", scaler_path="scaler.pkl", epochs=25, batch_size=64):
    model = build_alstm_model(input_shape=(X_train.shape[1], X_train.shape[2]),
                                   mem_dim=M_train.shape[1],
                                   seq_len=X_train.shape[1])

     # Optimizer with gradient clipping and lower lr
    opt = Adam(learning_rate=1e-4, clipnorm=1.0)

    # Use Huber loss (robust to outliers)
    model.compile(optimizer=opt, loss=Huber(delta=1.0), metrics=['mae'])

    base = os.path.splitext(model_path)[0]          # strip any extension
    ckpt_path = base + ".best.keras"

    # Callbacks: early stopping and reduce LR on plateau + checkpoint best model
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    history = model.fit(
        [X_train, M_train], y_train,
        validation_data=([X_val, M_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1
    )
    # Save model in Keras native format and also save HDF5 weights for compatibility:
    keras_path = base + ".keras"
    h5_weights_path = base + ".h5"

    model.save(keras_path)             # native Keras format
    try:
        model.save(h5_weights_path)    # attempt HDF5 save (may require h5py)
    except Exception:
        # fallback: save only weights if full h5 save fails
        model.save_weights(h5_weights_path)
    joblib.dump(scaler, scaler_path)
    print(f"✓ Model saved (native .keras) to: {keras_path}")
    print(f"✓ (Also saved weights/compat .h5 to: {h5_weights_path})")
    print(f"✓ Scaler saved to {scaler_path}")

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    # Evaluate
    y_pred = model.predict([X_val, M_val])
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation MAE (scaled): {mae:.4f}")
    print(f"Validation R²: {r2:.4f}")

    # Embeddings
    embedding_model = Model(inputs=model.inputs, outputs=model.get_layer('embedding').output)
    embeddings = embedding_model.predict([X_val, M_val])
    np.save("val_embeddings.npy", embeddings)
    print(f"✓ Saved embeddings to val_embeddings.npy")

    return model

# =========================================================
# 5. LOAD MODEL FOR FUTURE PREDICTIONS
# =========================================================
def load_trained_model(model_path="gas_alstm_mem.h5", scaler_path="scaler.pkl"):
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded model and scaler")
    return model, scaler

# =========================================================
# 6. PREDICT FUTURE GAS PRICES (Next N steps) — with memory
# =========================================================
def predict_future(model, recent_seq, recent_mem_val, scaler, steps=10):
    """
    recent_seq: shape (seq_len, d_features) scaled
    recent_mem_val: scalar mem_same_slot value (not scaled) for the prediction timestep
    scaler: fitted StandardScaler used earlier
    """
    input_seq = recent_seq.copy()  # scaled
    # scale mem val using scaler: scaler expects full-feature vector; we only need mem col index
    # We find the index of mem column as last feature used while fitting
    mem_col_idx = scaler.mean_.shape[0] - 1  # mem was last column when fitting
    # To scale mem_val, build a dummy vector with mem in last position and zeros elsewhere
    def scale_mem_scalar(raw_mem):
        dummy = np.zeros((1, scaler.mean_.shape[0]))
        dummy[0, mem_col_idx] = raw_mem
        scaled_dummy = scaler.transform(dummy)
        return scaled_dummy[0, mem_col_idx]

    predictions = []
    seq_len = input_seq.shape[0]
    for step in range(steps):
        # get scaled mem for this prediction (use same-time mem; fallback to mean if None)
        if recent_mem_val is None:
            mem_raw = scaler.mean_[mem_col_idx]  # use global mean (approx)
        else:
            mem_raw = recent_mem_val
        mem_scaled = scale_mem_scalar(mem_raw)

        pred_scaled = model.predict([input_seq[np.newaxis, :, :], np.array([[mem_scaled]])])[0, 0]
        predictions.append(pred_scaled)

        # Build next input row: shift left and append new row where gas_price replaced by predicted value
        new_row = input_seq[-1].copy()
        new_row[0] = pred_scaled  # gas_price (scaled)
        # optionally update mem_congestion feature or other features (here we simply keep last)
        input_seq = np.vstack([input_seq[1:], new_row])

    predictions = np.array(predictions).reshape(-1, 1)

    # Inverse transform: we need to place predictions into full-width array and inverse_transform
    pad = np.zeros((predictions.shape[0], scaler.mean_.shape[0] - 1))
    full_scaled = np.hstack([predictions, pad])  # gas_price at column 0, mem at last col etc.
    inv_preds = scaler.inverse_transform(full_scaled)[:, 0]  # gas price in original scale (Gwei)

    print("\n🔮 Future Predictions (Next {} Time Steps):".format(steps))
    for i, val in enumerate(inv_preds, 1):
        val = val*10
        print(f"Step +{i}: Predicted Gas Price = {val:.4f} Gwei")

    return inv_preds

# =========================================================
# 7. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    MONGO_URI = "mongodb://localhost:27017"
    DB_NAME = "eth_transactions_db"
    COLL_NAME = "decoded_transactions"

    df = load_data(MONGO_URI, DB_NAME, COLL_NAME, window='15min', mem_len=3)
    X_train, M_train, y_train, X_val, M_val, y_val, scaler = prepare_mem_data(df, seq_len=30)

    np.save("y_val.npy", y_val)
    print(f"✓ Saved y_val to y_val.npy")

    if not os.path.exists("gas_alstm_mem.h5"):
        model = train_and_save_model(X_train, M_train, y_train, X_val, M_val, y_val, scaler,
                                     model_path="gas_alstm_mem.h5", scaler_path="scaler.pkl", epochs=25)
    else:
        model, scaler = load_trained_model("gas_alstm_mem.h5", "scaler.pkl")

    # For prediction: use last sequence from validation + its mem_same_slot raw value (unscaled)
    recent_seq = X_val[-1]  # already scaled
    # Need raw mem value (unscaled) for the last target row (we can extract it from df)
    # We saved mem as last column when building scaled_all; index of last validation target in df:
    last_target_idx = len(df) - len(X_val)  # approximate; careful if sizes differ
    # Simpler: use the raw mem_same_slot of last timestamp
    recent_mem_raw = df["mem_same_slot"].iloc[-1] if "mem_same_slot" in df.columns else None

    predict_future(model, recent_seq, recent_mem_raw, scaler, steps=10)
