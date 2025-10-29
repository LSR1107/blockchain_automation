# =========================================================
# BTC_ALSTM_results.py — Backend integration for Bitcoin ALSTM
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from Analysis_backend.BTC_time_analysis import (
    load_bitcoin_data,
    add_memory_congestion_bitcoin,
    prepare_data_bitcoin,
    build_alstm_model,
)

# Ensure static folder exists
STATIC_DIR = os.path.join("backend", "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def run_btc_alstm_analysis():
    """
    Run Bitcoin ALSTM transaction fee prediction analysis and return summary.
    """
    try:
        # 1️⃣ MongoDB Configuration
        MONGO_URI = "mongodb://localhost:27017"
        DB_NAME = "btc_transactions_db"
        COLL_NAME = "decoded_transactions"

        # 2️⃣ Load and preprocess data
        df = load_bitcoin_data(MONGO_URI, DB_NAME, COLL_NAME)
        df = add_memory_congestion_bitcoin(df, roll_window=20)
        X_train, y_train, X_val, y_val, scaler = prepare_data_bitcoin(df, seq_len=30)

        model_path = "btc_fee_alstm.h5"
        scaler_path = "btc_scaler.pkl"

        # 3️⃣ Load existing model or train new one
        if not os.path.exists(model_path):
            print("⚠️ Model not found, training new ALSTM model...")
            from Analysis_backend.BTC_time_analysis import train_and_save_model
            model, _ = train_and_save_model(X_train, y_train, X_val, y_val, scaler)
        else:
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)

        # 4️⃣ Predict next 10 future steps
        recent_seq = X_val[-1]
        preds_scaled = []
        steps = 10
        input_seq = recent_seq.copy()

        for _ in range(steps):
            pred_scaled = model.predict(input_seq[np.newaxis, :, :])[0, 0]
            preds_scaled.append(pred_scaled)
            new_row = input_seq[-1].copy()
            new_row[0] = pred_scaled
            input_seq = np.vstack([input_seq[1:], new_row])

        preds_scaled = np.array(preds_scaled).reshape(-1, 1)
        pad = np.zeros((preds_scaled.shape[0], scaler.scale_.shape[0] - 1))
        full_scaled = np.hstack([preds_scaled, pad])
        inv_preds = scaler.inverse_transform(full_scaled)[:, 0]

        # 5️⃣ Evaluate model performance
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 6️⃣ Plot actual vs predicted (last 50 samples)
        plt.figure(figsize=(8, 4))
        plt.plot(y_val[-50:], label="Actual Fee (scaled)", color="blue")
        plt.plot(y_pred[-50:], label="Predicted Fee (scaled)", color="orange")
        plt.legend()
        plt.title("Bitcoin Transaction Fee Prediction (ALSTM)")
        plot_path = os.path.join(STATIC_DIR, "btc_alstm_plot.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        # 7️⃣ Return structured response
        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "records_used": len(df),
            "metrics": {
                "MAE": round(mae, 4),
                "R2_Score": round(r2, 4)
            },
            "predictions_next_10": inv_preds.tolist(),
            "plot_url": f"/static/btc_alstm_plot.png"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
