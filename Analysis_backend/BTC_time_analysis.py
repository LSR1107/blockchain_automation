
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.models import Model, load_model
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------
# 1. LOAD DATA FROM MONGODB
# ----------------------------
def load_bitcoin_data(mongo_uri, db_name, coll_name):
    print("="*60)
    print("LOADING BITCOIN DATA FROM MONGODB")
    print("="*60)
    client = MongoClient(mongo_uri)
    coll = client[db_name][coll_name]
    docs = list(coll.find({}).sort("block_number", 1))
    df = pd.DataFrame(docs)

    if df.empty:
        raise ValueError("No documents found in collection.")

    # parse block_time -> timestamp (like ETH version uses block_time)
    if "block_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["block_time"], utc=True)
    elif "block_time_iso" in df.columns:
        df["timestamp"] = pd.to_datetime(df["block_time_iso"], utc=True)
    else:
        raise ValueError("No 'block_time' or 'block_time_iso' column found in Bitcoin collection")

    # Ensure numeric columns exist; fill missing safely
    for col in ["fee_btc", "total_input_btc", "total_output_btc"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✓ Loaded {len(df):,} records ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df

# ----------------------------
# 2. SIMPLE MEMORY CONGESTION
# ----------------------------
def add_memory_congestion_bitcoin(df, roll_window=20):
    """Simple rolling mean memory congestion (0..scaled later by scaler)."""
    df = df.copy()
    # compute rolling mean of fee_btc (like ETH used rolling gas price)
    df["memory_congestion"] = df["fee_btc"].rolling(window=roll_window, min_periods=1).mean().fillna(method="bfill")
    print(f"✓ memory_congestion added (rolling window={roll_window})")
    return df

# ----------------------------
# 3. PREPARE SEQUENCES
# ----------------------------
def prepare_data_bitcoin(df, seq_len=30):
    """
    Prepare sequences for ALSTM. Mirrors ETH prepare_data:
    - features: ['fee_btc', 'total_input_btc', 'total_output_btc', 'memory_congestion']
    - target: next window fee_btc (first feature)
    Returns: X_train, y_train, X_val, y_val, scaler
    """
    feature_cols = ["fee_btc", "total_input_btc", "total_output_btc", "memory_congestion"]
    df_feat = df[feature_cols].fillna(0.0).reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_feat.values)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len, 0])   # predict next fee_btc

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"✓ Prepared sequences: total={len(X)}, train={len(X_train)}, val={len(X_val)}")
    return X_train, y_train, X_val, y_val, scaler

# ----------------------------
# 4. BUILD ATTENTION-LSTM
# ----------------------------
def build_alstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(32, return_sequences=True)(x)

    # Attention (self)
    attn = Attention()([x, x])
    x = Concatenate()([x, attn])

    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.2)(x)
    embedding = Dense(16, activation="relu", name="embedding")(x)
    outputs = Dense(1, activation="linear", name="fee_output")(embedding)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()
    return model

# ----------------------------
# 5. TRAIN & SAVE MODEL
# ----------------------------
def train_and_save_model(X_train, y_train, X_val, y_val, scaler,
                         model_path="btc_fee_alstm.h5", scaler_path="btc_scaler.pkl",
                         epochs=25, batch_size=32):
    model = build_alstm_model((X_train.shape[1], X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Save model and scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")

    # Evaluate & plot
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation MAE (scaled target): {mae:.6f}")
    print(f"Validation R² (scaled target): {r2:.4f}")

    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend(); plt.title("Training vs Validation Loss"); plt.show()

    # Save embeddings for downstream (GNN / RL) use
    embedding_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
    embeddings = embedding_model.predict(X_val)
    np.save("btc_val_embeddings.npy", embeddings)
    print("✓ Saved validation embeddings to btc_val_embeddings.npy")

    return model, history

# ----------------------------
# 6. LOAD MODEL & SCALER
# ----------------------------
def load_trained_model(model_path="btc_fee_alstm.h5", scaler_path="btc_scaler.pkl"):
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    print("✓ Loaded model and scaler")
    return model, scaler

# ----------------------------
# 7. FUTURE PREDICTION (iterative)
# ----------------------------
def predict_future(model, recent_seq_scaled, scaler, steps=10):
    """
    recent_seq_scaled: 2D array shape (seq_len, n_features) scaled by the scaler
    returns: inverse-transformed predicted fee_btc values (in BTC)
    """
    seq = recent_seq_scaled.copy()
    preds_scaled = []
    for _ in range(steps):
        s_in = seq[np.newaxis, :, :]  # shape (1, seq_len, features)
        pred_s = model.predict(s_in, verbose=0)[0, 0]  # scaled prediction (since training on scaled)
        preds_scaled.append(pred_s)

        # create new step: keep other features same as last row, replace fee feature with pred_s
        next_row = seq[-1].copy()
        next_row[0] = pred_s
        seq = np.vstack([seq[1:], next_row])

    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    # pad zeros for other features to inverse transform (scaler expects full feature dim)
    n_features = scaler.scale_.shape[0]
    pad = np.zeros((preds_scaled.shape[0], n_features - 1))
    full = np.hstack([preds_scaled, pad])
    inv = scaler.inverse_transform(full)[:, 0]
    return inv

# ----------------------------
# 8. MAIN
# ----------------------------
if __name__ == "__main__":
    # Config
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    DB = "btc_transactions_db"          # change if required
    COLL = "decoded_transactions"
    SEQ_LEN = 30
    EPOCHS = 25
    BATCH = 32
    FUTURE_STEPS = 10
    MODEL_FILE = "btc_fee_alstm.h5"
    SCALER_FILE = "btc_scaler.pkl"

    # 1. load
    df = load_bitcoin_data(MONGO_URI, DB, COLL)

    # 2. add memory congestion (rolling mean)
    df = add_memory_congestion_bitcoin(df, roll_window=20)

    # 3. prepare sequences
    X_train, y_train, X_val, y_val, scaler = prepare_data_bitcoin(df, seq_len=SEQ_LEN)

    # save y_val for later analysis (scaled space)
    np.save("btc_y_val.npy", y_val)
    print("✓ Saved y_val to btc_y_val.npy")

    # 4. train (or load)
    if not os.path.exists(MODEL_FILE):
        model, history = train_and_save_model(X_train, y_train, X_val, y_val, scaler,
                                             model_path=MODEL_FILE, scaler_path=SCALER_FILE,
                                             epochs=EPOCHS, batch_size=BATCH)
    else:
        model, scaler = load_trained_model(MODEL_FILE, SCALER_FILE)

    # 5. predict next N steps using last validation sequence
    recent_seq_scaled = X_val[-1]  # already scaled
    preds = predict_future(model, recent_seq_scaled, scaler, steps=FUTURE_STEPS)

    print("\n🔮 Future Predictions (fee_btc):")
    for i, v in enumerate(preds, 1):
        print(f"Step +{i}: Predicted fee_btc = {v:.8f} BTC")

    # optionally save predictions to Mongo / file if needed
    np.save("btc_future_preds.npy", preds)
    print("✓ Saved future predictions to btc_future_preds.npy")
