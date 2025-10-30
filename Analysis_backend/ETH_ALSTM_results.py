# =========================================================
# ETH_ALSTM_results.py — Backend integration for Ethereum ALSTM
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pymongo import MongoClient
from tensorflow.keras.models import load_model
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
from Analysis_backend.ETH_time_analysis import load_data, prepare_data, build_alstm_model  # import from your existing module

# Ensure static folder exists
STATIC_DIR = os.path.join("backend", "static")
os.makedirs(STATIC_DIR, exist_ok=True)


def run_eth_alstm_analysis():
    """
    Run Ethereum ALSTM gas price prediction analysis and return summary.
    """
    try:
        MONGO_URI = "mongodb://localhost:27017"
        DB_NAME = "eth_transactions_db"
        COLL_NAME = "decoded_transactions"

        # 1️⃣ Load Data
        df = load_data(MONGO_URI, DB_NAME, COLL_NAME)
        X_train, y_train, X_val, y_val, scaler = prepare_data(df)

        model_path = "gas_alstm.h5"
        scaler_path = "scaler.pkl"

        # 2️⃣ Load existing model
        if not os.path.exists(model_path):
            print("⚠️ Model not found, training new ALSTM...")
            from Analysis_backend.ETH_time_analysis import train_and_save_model
            model = train_and_save_model(X_train, y_train, X_val, y_val, scaler)
        else:
            model = load_model(model_path, compile=False)
            scaler = joblib.load(scaler_path)

        # 3️⃣ Predict
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

        # 4️⃣ Compute metrics
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # 5️⃣ Plot and save prediction
        plt.figure(figsize=(8, 4))
        plt.plot(y_val[-50:], label="Actual Gas Price", color="blue")
        plt.plot(y_pred[-50:], label="Predicted Gas Price", color="orange")
        plt.legend()
        plt.title("Ethereum Gas Price Prediction (ALSTM)")
        plot_path = os.path.join(STATIC_DIR, "eth_alstm_plot.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "records_used": len(df),
            "metrics": {
                "MAE": round(mae, 4),
                "R2_Score": round(r2, 4)
            },
            "predictions_next_10": inv_preds.tolist(),
            "plot_url": f"/static/eth_alstm_plot.png"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
