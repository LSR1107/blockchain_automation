import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend for server/macOS
import matplotlib.pyplot as plt
import os, pickle

from sklearn.linear_model import LinearRegression
from Analysis_backend import BTC_graph_analysis as notebook  # your notebook file


# =================== CORE PREDICTION FUNCTION ===================
def predict_with_probabilities(model, block_graphs, tx_graphs, device='cpu', slots=None):
    model.eval()
    model = model.to(device)
    predictions = []

    with torch.no_grad():
        for i, (block_data, tx_data) in enumerate(zip(block_graphs, tx_graphs)):
            block_data = block_data.to(device)
            tx_data = tx_data.to(device)

            if not hasattr(block_data, 'batch'):
                block_data.batch = torch.zeros(block_data.x.size(0), dtype=torch.long, device=device)
            if not hasattr(tx_data, 'batch'):
                tx_data.batch = torch.zeros(tx_data.x.size(0), dtype=torch.long, device=device)

            logits = model(block_data, tx_data)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()

            predictions.append({
                'step': i + 1,
                'slot': slots[i] if slots else None,
                'predicted_class': pred_class,
                'prob_low': probs[0, 0].item(),
                'prob_high': probs[0, 1].item(),
                'confidence': float(max(probs[0, 0], probs[0, 1])),
                'prediction_strength': abs(logits[0, 1] - logits[0, 0]).item()
            })

    return pd.DataFrame(predictions)


# =================== CALIBRATION ===================
def calibrate_probabilities_to_utilization(model, block_graphs, tx_graphs, true_utilizations, device='cpu'):
    model.eval()
    predicted_probs = []

    with torch.no_grad():
        for block_data, tx_data in zip(block_graphs, tx_graphs):
            block_data = block_data.to(device)
            tx_data = tx_data.to(device)

            if not hasattr(block_data, 'batch'):
                block_data.batch = torch.zeros(block_data.x.size(0), dtype=torch.long, device=device)
            if not hasattr(tx_data, 'batch'):
                tx_data.batch = torch.zeros(tx_data.x.size(0), dtype=torch.long, device=device)

            logits = model(block_data, tx_data)
            probs = F.softmax(logits, dim=1)
            predicted_probs.append(probs[0, 1].item())

    reg = LinearRegression()
    reg.fit(np.array(predicted_probs).reshape(-1, 1), true_utilizations)

    def prob_to_utilization(prob_high):
        util = reg.predict(np.array([[prob_high]]))[0]
        return np.clip(util, 0, 1)

    return prob_to_utilization, reg


# =================== PREDICTION + UTILIZATION ===================
def predict_congestion_with_scores(model, block_graphs, tx_graphs, prob_to_util_fn, device='cpu', util_threshold=0.7, slots=None):
    df = predict_with_probabilities(model, block_graphs, tx_graphs, device, slots)
    df['estimated_utilization'] = df['prob_high'].apply(prob_to_util_fn)
    df['estimated_utilization_pct'] = df['estimated_utilization'] * 100
    df['congestion_status'] = df['predicted_class'].map({0: 'LOW', 1: 'HIGH'})

    def severity(u):
        if u < 0.4: return '🟢 LIGHT'
        elif u < 0.65: return '🟡 MODERATE'
        elif u < 0.80: return '🟠 HIGH'
        else: return '🔴 CRITICAL'
    df['severity'] = df['estimated_utilization'].apply(severity)
    return df


# =================== VISUALIZATION ===================
def visualize_probability_analysis(predictions_df, threshold=0.7, save_path="backend/static/btc_gnn_forecast.png"):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Probabilities
    axes[0].plot(predictions_df['step'], predictions_df['prob_high'], marker='o', color='#ff6b6b', label='P(High)')
    axes[0].plot(predictions_df['step'], predictions_df['prob_low'], marker='s', color='#51cf66', label='P(Low)')
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.6)
    axes[0].legend(); axes[0].set_title('Bitcoin Congestion Probabilities'); axes[0].grid(True, alpha=0.3)

    # Utilization
    axes[1].plot(predictions_df['step'], predictions_df['estimated_utilization_pct'], color='#4c6ef5', marker='D')
    axes[1].axhline(threshold*100, color='red', linestyle='--', label=f'Threshold {threshold*100:.0f}%')
    axes[1].legend(); axes[1].set_title('Estimated Block Utilization (BTC)'); axes[1].grid(True, alpha=0.3)

    # Confidence
    colors = ['#51cf66' if c == 0 else '#ff6b6b' for c in predictions_df['predicted_class']]
    axes[2].bar(predictions_df['step'], predictions_df['confidence']*100, color=colors, alpha=0.8)
    axes[2].set_title('Prediction Confidence (%)'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


# =================== MAIN WORKFLOW ===================
def complete_congestion_analysis(model, block_graphs, tx_graphs, block_stats, device='cpu', future_steps=10):
    true_utilizations = block_stats['block_utilization'].values[-len(block_graphs):]
    prob_to_util_fn, _ = calibrate_probabilities_to_utilization(model, block_graphs, tx_graphs, true_utilizations, device)

    future_blocks = block_graphs[-future_steps:]
    future_txs = tx_graphs[-future_steps:]
    future_slots = [g.slot for g in future_blocks] if hasattr(future_blocks[0], 'slot') else None

    predictions_df = predict_congestion_with_scores(model, future_blocks, future_txs, prob_to_util_fn, device, slots=future_slots)
    plot_path = visualize_probability_analysis(predictions_df)
    return predictions_df, prob_to_util_fn, plot_path


# =================== LOAD SAVED GRAPHS ===================
def load_data(filename='Analysis_backend/data_btc.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# =================== ENTRYPOINT ===================
def run_btc_gnn_analysis(future_steps=10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block_graphs, tx_graphs, labels, block_stats = load_data()

    df = notebook.load_btc_transactions()
    block_stats = notebook.build_block_stats_btc(df)

    model = notebook.HybridCongestionModel(
        block_in=block_graphs[0].x.shape[1],
        tx_in=tx_graphs[0].x.shape[1],
        hidden=64, out_dim=2, dropout=0.3
    )
    model.load_state_dict(torch.load('Analysis_backend/saved_models/best_btc_model.pth', map_location=device))
    model.to(device).eval()

    predictions_df, calibration_fn, plot_path = complete_congestion_analysis(
        model, block_graphs, tx_graphs, block_stats, device, future_steps
    )

    summary = {
        "avg_prob_high": float(predictions_df['prob_high'].mean()),
        "avg_utilization": float(predictions_df['estimated_utilization_pct'].mean()),
        "peak_utilization": float(predictions_df['estimated_utilization_pct'].max()),
        "high_congestion_blocks": int((predictions_df['predicted_class'] == 1).sum()),
        "total_blocks": len(predictions_df),
        "plot_url": f"/static/{os.path.basename(plot_path)}"
    }

    return {
        "summary": summary,
        "predictions": predictions_df.to_dict(orient='records')
    }
