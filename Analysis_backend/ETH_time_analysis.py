# =========================================================
# GAS PRICE PREDICTION WITH ATTENTION-LSTM
# =========================================================


import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate
from tensorflow.keras.models import Model, load_model
import joblib
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. LOAD DATA FROM MONGODB
# =========================================================
def load_data(mongo_uri, db_name, coll_name):
    print("="*60)
    print("LOADING DATA FROM MONGODB")
    print("="*60)
    client = MongoClient(mongo_uri)
    coll = client[db_name][coll_name]
    
    docs = list(coll.find({}).sort("block_number", 1))
    df = pd.DataFrame(docs)
    
    df["timestamp"] = pd.to_datetime(df["block_time"], utc=True)
    
    # Convert Wei to Gwei
    for col in ["gas_price", "max_fee_per_gas", "max_priority_fee_per_gas"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) / 1e9

    # Simple congestion factor (mocked)
    df["memory_congestion"] = df["gas_price"].rolling(20).mean().fillna(method='bfill')
    
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"✓ Loaded {len(df):,} records ({df['timestamp'].min()} → {df['timestamp'].max()})")
    return df

# =========================================================
# 2. PREPARE SEQUENCES
# =========================================================
def prepare_data(df, seq_len=30):
    feature_cols = ["gas_price", "max_fee_per_gas", "max_priority_fee_per_gas", "memory_congestion"]
    df = df[feature_cols].fillna(0)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(df) - seq_len):
        X.append(scaled_data[i:i+seq_len])
        y.append(scaled_data[i+seq_len, 0])  # Predict next gas_price

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:], scaler

# =========================================================
# 3. BUILD ATTENTION-LSTM MODEL
# =========================================================
def build_alstm_model(input_shape):
    inputs = Input(shape=input_shape)

    x = LSTM(64, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    attn_out = Attention()([x, x])
    x = Concatenate()([x, attn_out])

    x = LSTM(32, return_sequences=False)(x)
    embedding = Dense(16, activation='relu', name='embedding')(x)
    outputs = Dense(1)(embedding)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# =========================================================
# 4. TRAIN & SAVE MODEL
# =========================================================
def train_and_save_model(X_train, y_train, X_val, y_val, scaler, model_path="gas_alstm.h5", scaler_path="scaler.pkl"):
    model = build_alstm_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=32,
        verbose=1
    )

    # Save model & scaler
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✓ Model saved to {model_path}")
    print(f"✓ Scaler saved to {scaler_path}")

    # Plot training history
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.show()

    # Evaluate
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R²: {r2:.4f}")

    # Get final embeddings
    embedding_model = Model(inputs=model.input, outputs=model.get_layer('embedding').output)
    embeddings = embedding_model.predict(X_val)
    np.save("val_embeddings.npy", embeddings)
    print(f"✓ Saved embeddings to val_embeddings.npy")

    


    return model

# =========================================================
# 5. LOAD MODEL FOR FUTURE PREDICTIONS
# =========================================================
def load_trained_model(model_path="gas_alstm.h5", scaler_path="scaler.pkl"):
    model = load_model(model_path, compile=False)
    scaler = joblib.load(scaler_path)
    print(f"✓ Loaded model and scaler")
    return model, scaler

# =========================================================
# 6. PREDICT FUTURE GAS PRICES (Next 10 steps)
# =========================================================
def predict_future(model, recent_data, scaler, steps=10):
    input_seq = recent_data.copy()
    predictions = []

    for _ in range(steps):
        pred_scaled = model.predict(input_seq[np.newaxis, :, :])[0, 0]
        predictions.append(pred_scaled)

        new_row = input_seq[-1].copy()
        new_row[0] = pred_scaled  # update gas_price with predicted
        input_seq = np.vstack([input_seq[1:], new_row])

    predictions = np.array(predictions).reshape(-1, 1)
    pad = np.zeros((predictions.shape[0], scaler.scale_.shape[0] - 1))
    full_scaled = np.hstack([predictions, pad])
    inv_preds = scaler.inverse_transform(full_scaled)[:, 0]

    print("\n🔮 Future Predictions (Next 10 Time Steps):")
    for i, val in enumerate(inv_preds, 1):
        print(f"Step +{i}: Predicted Gas Price = {val:.4f} Gwei")

    return inv_preds

# =========================================================
# 7. MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    MONGO_URI = "mongodb://localhost:27017"
    DB_NAME = "eth_transactions_db"
    COLL_NAME = "decoded_transactions"

    df = load_data(MONGO_URI, DB_NAME, COLL_NAME)
    X_train, y_train, X_val, y_val, scaler = prepare_data(df)

    np.save("y_val.npy", y_val)
    print(f"✓ Saved y_val to y_val.npy")

    # Train only if model doesn't exist
    if not os.path.exists("gas_alstm.h5"):
        model = train_and_save_model(X_train, y_train, X_val, y_val, scaler)
    else:
        model, scaler = load_trained_model()

    # Predict next 10 steps using last sequence
    recent_seq = X_val[-1]
    predict_future(model, recent_seq, scaler, steps=10)
