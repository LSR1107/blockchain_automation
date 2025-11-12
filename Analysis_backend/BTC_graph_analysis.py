#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# In[2]:


def load_btc_transactions(
    mongo_uri="mongodb://localhost:27017/",
    db_name="btc_transactions_db",
    collection_name="decoded_transactions"
):
    """
    Load Bitcoin transactions from MongoDB with enhanced feature extraction.
    Compatible with Bitcoin blockchain structure.
    Handles missing fee_rate and derives it from fee/size.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Pull documents with all relevant fields
    docs = list(collection.find({}, {
        "block_number": 1,
        "block_time": 1,
        "tx_hash": 1,
        "inputs": 1,
        "outputs": 1,
        "fee_btc": 1,
        "total_input_btc": 1,
        "total_output_btc": 1,
        "size": 1,
        "status": 1
    }))
    
    df = pd.DataFrame(docs)
    
    if df.empty:
        print("⚠️ No Bitcoin transactions found in MongoDB.")
        return df
    
    print(f"Loaded {len(df)} transactions from MongoDB")
    
    # Rename to standardized names
    df.rename(columns={
        "block_number": "slot",
        "block_time": "blockTime",
        "tx_hash": "signature",
        "fee_btc": "fee",
        "total_input_btc": "input_total",
        "total_output_btc": "output_total"
    }, inplace=True)
    
    # Extract input and output counts from arrays
    df["input_count"] = df["inputs"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["output_count"] = df["outputs"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    
    # Calculate transaction size (approximation for Bitcoin)
    # Size ≈ (input_count × 148) + (output_count × 34) + 10
    # This is based on typical Bitcoin transaction structure
    # 148 bytes per input (legacy), 34 bytes per output, ~10 bytes overhead
    df["size"] = (df["input_count"] * 148) + (df["output_count"] * 34) + 10
    
    # Ensure size is at least 100 bytes (minimum viable transaction)
    df["size"] = df["size"].clip(lower=100)
    
    # Verify critical columns exist
    critical_cols = ['slot', 'blockTime', 'input_count', 'output_count', 'input_total', 'output_total', 'fee', 'size']
    missing = [col for col in critical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"❌ Critical columns missing: {missing}\n"
                        f"Available columns: {list(df.columns)}")
    
    # Convert to datetime
    if "blockTime" in df.columns:
        df["blockTime"] = pd.to_datetime(df["blockTime"], errors="coerce")
    
    # Ensure all numeric fields exist and are clean
    numeric_cols = [
        "input_count", "output_count", "input_total", "output_total",
        "fee", "fee_rate", "size", "vsize"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # DEBUG: Check fee values before conversion
    print(f"\n🔍 DEBUG - Fee Statistics BEFORE conversion:")
    print(f"   Fee min: {df['fee'].min()}, Fee max: {df['fee'].max()}, Fee mean: {df['fee'].mean()}")
    print(f"   Sample fees (first 5): {df['fee'].head().tolist()}")
    
    # Convert fee from BTC to satoshis (1 BTC = 100,000,000 satoshis)
    # If fee is already in satoshis, this will multiply by 1e8, making it huge
    # Check if fees are suspiciously small (< 0.0001) - likely in BTC
    if df['fee'].max() < 0.01:  # Likely in BTC format
        print("   → Converting fee from BTC to satoshis (×100,000,000)")
        df['fee'] = df['fee'] * 1e8
    else:
        print("   → Fee already appears to be in satoshis")
    
    print(f"\n🔍 DEBUG - Fee Statistics AFTER conversion:")
    print(f"   Fee min: {df['fee'].min()}, Fee max: {df['fee'].max()}, Fee mean: {df['fee'].mean()}")
    
    # Ensure boolean columns
    if "is_coinbase" not in df.columns:
        df["is_coinbase"] = 0
    df["is_coinbase"] = df["is_coinbase"].astype(int)
    
    # Derive fee_rate if missing: sat/byte
    df["fee_rate"] = np.where(
        (df["fee_rate"] == 0) & (df["size"] > 0),
        (df["fee"] / df["size"]),  # Calculate from fee/size
        df["fee_rate"]
    )
    
    # Use vsize (virtual size) if available, otherwise size
    df["tx_size"] = np.where(df["vsize"] > 0, df["vsize"], df["size"])
    df["tx_size"] = np.where(df["tx_size"] == 0, 250, df["tx_size"])  # avg tx size ~250 bytes
    
    # Derived features
    # Value efficiency: output value per byte
    df["value_per_byte"] = df["output_total"] / (df["tx_size"] + 1e-9)
    
    # Transaction complexity: average inputs/outputs
    df["io_count"] = (df["input_count"] + df["output_count"]) / 2
    
    # Input diversity: higher with more inputs (simplified proxy)
    df["input_diversity"] = np.log1p(df["input_count"])
    
    # Output diversity: higher with more outputs
    df["output_diversity"] = np.log1p(df["output_count"])
    
    # Average value per input (indicates transaction importance)
    df["value_per_input"] = df["input_total"] / (df["input_count"] + 1e-9)
    
    # Average output value
    df["avg_output_value"] = df["output_total"] / (df["output_count"] + 1e-9)
    
    # Filter out coinbase transactions for later graph construction
    df["is_regular"] = 1 - df["is_coinbase"]
    
    # Sort chronologically for later graph windows
    df = df.sort_values("blockTime").reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} Bitcoin transactions across {df['slot'].nunique()} blocks.")
    print(f"   Regular (non-coinbase) transactions: {df['is_regular'].sum()}")
    print(f"   Coinbase transactions: {df['is_coinbase'].sum()}")
    print(f"   Fee rate range: {df['fee_rate'].min():.2f} - {df['fee_rate'].max():.2f} sat/byte")
    print(f"   Block time range: {df['blockTime'].min()} to {df['blockTime'].max()}")
    
    return df


# In[3]:


# =================== BLOCK STATISTICS ===================
def build_block_stats_btc(transactions_df):
    """
    Build block-level statistics for Bitcoin
    Compatible with MongoDB-extracted data where columns are:
    slot (block_height), blockTime, fee_rate, tx_size, input_count, output_count, etc.
    """
    
    if transactions_df.empty:
        return pd.DataFrame()
    
    df = transactions_df.copy()
    
    # Ensure expected columns exist
    required_cols = ["slot", "blockTime", "fee_rate", "tx_size", "input_count", "output_count", "is_coinbase"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"\nProcessing {len(df)} transactions from {df['slot'].nunique()} blocks...")
    
    # Group by slot (block height)
    block_groups = df.groupby("slot")
    
    block_stats = []
    BITCOIN_BLOCK_LIMIT = 4_000_000  # 4MB in bytes
    
    for slot, group in block_groups:
        tx_count = len(group)
        regular_tx_count = len(group[group["is_coinbase"] == 0])
        coinbase_tx_count = len(group[group["is_coinbase"] == 1])
        
        # Block size metrics
        total_block_size = group["tx_size"].sum()
        block_utilization = total_block_size / BITCOIN_BLOCK_LIMIT
        
        # Fee rate metrics (sat/byte)
        regular_group = group[group["is_coinbase"] == 0]
        if len(regular_group) > 0:
            fee_rates = regular_group["fee_rate"]
            avg_fee_rate = fee_rates.mean()
            median_fee_rate = fee_rates.median()
            fee_variance = fee_rates.std() if len(fee_rates) > 1 else 0
            min_fee_rate = fee_rates.min()
            max_fee_rate = fee_rates.max()
        else:
            avg_fee_rate = median_fee_rate = fee_variance = min_fee_rate = max_fee_rate = 0
        
        # Value metrics
        if "output_total" in group.columns:
            values_btc = group["output_total"] / 1e8  # Convert satoshis to BTC
            avg_value = values_btc.mean()
            median_value = values_btc.median()
            total_value = values_btc.sum()
        else:
            avg_value = median_value = total_value = 0
        
        # Transaction complexity
        if len(regular_group) > 0:
            avg_input_count = regular_group["input_count"].mean()
            avg_output_count = regular_group["output_count"].mean()
        else:
            avg_input_count = avg_output_count = 0
        
        # Timestamp
        timestamp = group["blockTime"].iloc[0] if "blockTime" in group.columns else np.nan
        
        # Average transaction size
        avg_tx_size = group["tx_size"].mean() if len(group) > 0 else 0
        
        block_stats.append({
            "slot": slot,
            "block_number": slot,
            "timestamp": timestamp,
            "tx_count": tx_count,
            "regular_tx_count": regular_tx_count,
            "coinbase_tx_count": coinbase_tx_count,
            "total_block_size": total_block_size,
            "block_limit": BITCOIN_BLOCK_LIMIT,
            "block_utilization": block_utilization,
            "avg_fee_rate": avg_fee_rate,
            "median_fee_rate": median_fee_rate,
            "fee_variance": fee_variance,
            "min_fee_rate": min_fee_rate,
            "max_fee_rate": max_fee_rate,
            "avg_tx_size": avg_tx_size,
            "avg_input_count": avg_input_count,
            "avg_output_count": avg_output_count,
            "avg_value": avg_value,
            "median_value": median_value,
            "total_value": total_value,
        })
    
    bs = pd.DataFrame(block_stats).sort_values("slot").reset_index(drop=True)
    
    # TPS computation (transactions per second)
    bs["block_time_delta"] = bs["timestamp"].diff().dt.total_seconds().fillna(600)  # Bitcoin target: 600s
    bs["tps"] = bs["tx_count"] / bs["block_time_delta"]
    bs["tps"] = bs["tps"].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Normalization for congestion score
    bs["util_norm"] = bs["block_utilization"].clip(0, 1)
    bs["tps_norm"] = (bs["tps"] - bs["tps"].min()) / (bs["tps"].max() - bs["tps"].min() + 1e-8)
    bs["fee_norm"] = (bs["avg_fee_rate"] - bs["avg_fee_rate"].min()) / (
        bs["avg_fee_rate"].max() - bs["avg_fee_rate"].min() + 1e-8
    )
    
    # Composite congestion score (weighted)
    bs["congestion_score"] = (
        0.1 * bs["util_norm"] +
        0.5 * bs["tps_norm"] +
        0.4 * bs["fee_norm"]
    )
    
    print("\n✅ Bitcoin Block Statistics Built Successfully ✅")
    print(f"Total Blocks: {len(bs)}, Total Transactions: {bs['tx_count'].sum():,}")
    print(f"Avg Utilization: {bs['block_utilization'].mean():.4f}")
    print(f"Avg Fee Rate: {bs['avg_fee_rate'].mean():.2f} sat/byte")
    print(f"Avg TPS: {bs['tps'].mean():.4f}")
    print(f"Avg Congestion Score: {bs['congestion_score'].mean():.4f}")
    
    return bs


# In[4]:


# =================== GRAPH BUILDING ===================
def build_block_graphs_btc(block_stats, window=8, congestion_metric='composite', threshold=None):
    """
    Build block-level graphs with multiple congestion metrics for Bitcoin
    
    Args:
        block_stats: DataFrame from build_block_stats_btc()
        window: Sliding window size (blocks)
        congestion_metric: 'utilization', 'fee_rate', 'composite', or 'multi'
        threshold: Congestion threshold (auto-determined if None)
    """
    bs = block_stats.sort_values("slot").reset_index(drop=True).copy()
    
    # Select feature columns for Bitcoin
    feat_cols = [
        "tx_count", "regular_tx_count", "coinbase_tx_count",
        "avg_fee_rate", "median_fee_rate", "fee_variance",
        "block_time_delta", "block_utilization",
        "avg_tx_size", "avg_input_count", "avg_output_count",
        "tps", "congestion_score"
    ]
    
    # Add optional features
    optional_cols = ["min_fee_rate", "max_fee_rate", "avg_value", "median_value", "total_value"]
    for col in optional_cols:
        if col in bs.columns:
            feat_cols.append(col)
    
    X = bs[feat_cols].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    # Determine threshold automatically if not provided
    if threshold is None:
        if congestion_metric == 'utilization':
            threshold = bs['block_utilization'].quantile(0.60)
            metric_col = 'block_utilization'
        elif congestion_metric == 'fee_rate':
            threshold = bs['avg_fee_rate'].quantile(0.60)
            metric_col = 'avg_fee_rate'
        elif congestion_metric == 'composite':
            threshold = bs['congestion_score'].quantile(0.60)
            metric_col = 'congestion_score'
        else:  # multi
            util_thresh = bs['block_utilization'].quantile(0.70)
            fee_thresh = bs['avg_fee_rate'].quantile(0.70)
            metric_col = None
    else:
        if congestion_metric == 'utilization':
            metric_col = 'block_utilization'
        elif congestion_metric == 'fee_rate':
            metric_col = 'avg_fee_rate'
        else:
            metric_col = 'congestion_score'
    
    print(f"\n{'='*70}")
    print(f"Building block graphs with congestion_metric='{congestion_metric}'")
    if metric_col:
        print(f"Threshold: {threshold:.4f} on '{metric_col}'")
    print(f"{'='*70}\n")
    
    graphs, labels, target_slots = [], [], []
    
    for i in range(window - 1, len(bs)):
        x_win = torch.tensor(Xs[i - window + 1:i + 1], dtype=torch.float)
        n = x_win.size(0)
        
        if n > 1:
            src = list(range(n - 1))
            dst = list(range(1, n))
            edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Create label based on chosen metric
        if congestion_metric == 'multi':
            y = int((bs.loc[i, "block_utilization"] > util_thresh) or 
                   (bs.loc[i, "avg_fee_rate"] > fee_thresh))
        else:
            y = int(bs.loc[i, metric_col] > threshold)
        
        data = Data(x=x_win, edge_index=edge_index)
        data.slot = int(bs.loc[i, "slot"])
        
        graphs.append(data)
        labels.append(y)
        target_slots.append(data.slot)
    
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Print label distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label Distribution:")
    for label, count in zip(unique, counts):
        status = "NORMAL" if label == 0 else "CONGESTED"
        print(f"  Class {label} ({status}): {count} samples ({count/len(labels)*100:.1f}%)")
    
    return graphs, labels_tensor, target_slots



# In[5]:


def build_tx_graphs_btc(df, target_slots):
    """
    Build transaction-level graphs for each target slot (Bitcoin)
    """
    # Filter out coinbase transactions
    df_regular = df[df["is_coinbase"] == 0].copy()
    
    # Prepare features
    feat_cols = ["fee_rate", "input_count", "output_count", "tx_size", "value_per_byte", "input_diversity"]
    available_cols = [col for col in feat_cols if col in df_regular.columns]
    
    if len(available_cols) == 0:
        raise ValueError("No feature columns found in dataframe")
    
    X = df_regular[available_cols].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    # Group by slot
    slot2idxs = df_regular.groupby("slot").indices
    graphs = []
    
    for s in target_slots:
        idxs = list(slot2idxs.get(s, []))
        n = len(idxs)
        
        if n == 0:
            # Empty graph with single dummy node
            x = torch.zeros((1, len(available_cols)), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            x = torch.tensor(Xs[idxs], dtype=torch.float)
            
            if n > 1:
                # Create edges based on transaction order within block
                src = list(range(n - 1))
                dst = list(range(1, n))
                edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data.slot = int(s)
        graphs.append(data)
    
    return graphs


# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# =========================================================
# MODEL COMPONENTS
# =========================================================
class BlockEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(BlockEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
    
    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return global_mean_pool(h, batch)


class TxEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(TxEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
    
    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return global_mean_pool(h, batch)


class HybridCongestionModel(nn.Module):
    def __init__(self, block_in, tx_in, hidden=64, out_dim=2, dropout=0.3):
        super(HybridCongestionModel, self).__init__()
        self.block_encoder = BlockEncoder(block_in, hidden, hidden, dropout)
        self.tx_encoder = TxEncoder(tx_in, hidden, hidden, dropout)
        self.fc1 = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, out_dim)
    
    def forward(self, block_data, tx_data):
        block_emb = self.block_encoder(block_data.x, block_data.edge_index, block_data.batch)
        tx_emb = self.tx_encoder(tx_data.x, tx_data.edge_index, tx_data.batch)
        h = torch.cat([block_emb, tx_emb], dim=1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        return self.fc3(h)

# =========================================================
# DATA UTILITIES
# =========================================================
class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, block_graphs, tx_graphs, labels):
        self.block_graphs = block_graphs
        self.tx_graphs = tx_graphs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.block_graphs[idx], self.tx_graphs[idx], self.labels[idx]


def collate_fn(batch):
    block_batch, tx_batch, labels = zip(*batch)
    return (
        Batch.from_data_list(block_batch),
        Batch.from_data_list(tx_batch),
        torch.tensor(labels, dtype=torch.long)
    )


def split_dataset(block_graphs, tx_graphs, labels, test_size=0.25):
    if torch.is_tensor(labels):
        labels = labels.numpy()
    idx_train, idx_test = train_test_split(
        range(len(labels)), test_size=test_size, random_state=42,
        stratify=labels if len(np.unique(labels)) > 1 else None
    )
    return (
        [block_graphs[i] for i in idx_train],
        [block_graphs[i] for i in idx_test],
        [tx_graphs[i] for i in idx_train],
        [tx_graphs[i] for i in idx_test],
        labels[idx_train],
        labels[idx_test]
    )

# =========================================================
# TRAINING + VALIDATION
# =========================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for block_data, tx_data, labels in loader:
        block_data, tx_data, labels = block_data.to(device), tx_data.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(block_data, tx_data)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (out.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, preds, y_true = 0, 0, [], []
    with torch.no_grad():
        for block_data, tx_data, labels in loader:
            block_data, tx_data, labels = block_data.to(device), tx_data.to(device), labels.to(device)
            out = model(block_data, tx_data)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            preds.extend(out.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.cpu().numpy())
            correct += (out.argmax(dim=1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset), np.array(preds), np.array(y_true)


def run_training(block_graphs, tx_graphs, labels, model, epochs=20, batch_size=16, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_blocks, test_blocks, train_txs, test_txs, y_train, y_test = split_dataset(block_graphs, tx_graphs, labels)

    train_loader = torch.utils.data.DataLoader(HybridDataset(train_blocks, train_txs, y_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(HybridDataset(test_blocks, test_txs, y_test), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc, train_losses, val_losses = 0, [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, y_true = evaluate(model, test_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch:02d} | Train Loss={tr_loss:.4f} | Val Loss={val_loss:.4f} | Train Acc={tr_acc:.3f} | Val Acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("saved_models", exist_ok=True) 
            torch.save(model.state_dict(), "saved_models/best_btc_model.pth")

    print(f"✅ Best Validation Accuracy: {best_acc:.4f}")
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Save embeddings
    save_validation_embeddings(model, test_blocks, test_txs, y_test)
    return model


# =========================================================
# SAVE & LOAD UTILITIES
# =========================================================
def save_validation_embeddings(model, val_blocks, val_txs, y_val, device="cpu"):
    model.eval()
    model = model.to(device)
    loader = torch.utils.data.DataLoader(HybridDataset(val_blocks, val_txs, y_val), batch_size=16, shuffle=False, collate_fn=collate_fn)
    embeddings, labels = [], []
    with torch.no_grad():
        for block_data, tx_data, label in loader:
            block_data, tx_data = block_data.to(device), tx_data.to(device)
            block_emb = model.block_encoder(block_data.x, block_data.edge_index, block_data.batch)
            tx_emb = model.tx_encoder(tx_data.x, tx_data.edge_index, tx_data.batch)
            emb = torch.cat([block_emb, tx_emb], dim=1)
            embeddings.append(emb.cpu().numpy())
            labels.append(label.cpu().numpy())
    np.save("saved_outputs/val_embeddings.npy", np.concatenate(embeddings))
    np.save("saved_outputs/y_val_true.npy", np.concatenate(labels))
    print("✅ Validation embeddings & labels saved!")


# =========================================================
# FUTURE PREDICTION
# =========================================================
def predict_future_congestion(model_path, block_graphs, tx_graphs, future_steps=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCongestionModel(block_graphs[0].x.shape[1], tx_graphs[0].x.shape[1]).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(min(future_steps, len(block_graphs))):
            b = Batch.from_data_list([block_graphs[i]]).to(device)
            t = Batch.from_data_list([tx_graphs[i]]).to(device)
            out = model(b, t)
            preds.append(out.argmax(dim=1).item())
    print(f"✅ Future Predictions (next {future_steps}): {preds}")
    return preds


# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.data import Batch
import os

# =========================================================
# HELPER UTILITIES
# =========================================================
def calculate_metrics(y_true, preds):
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0)
    }


def analyze_predictions(model, block_graphs, tx_graphs, labels, device):
    """Get predictions and probabilities"""
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for b, t, y in zip(block_graphs, tx_graphs, labels):
            bb = Batch.from_data_list([b]).to(device)
            tt = Batch.from_data_list([t]).to(device)
            out = model(bb, tt)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().item()
            pred = out.argmax(dim=1).cpu().item()
            preds.append(pred)
            probs.append(prob)
    return np.array(preds), np.array(probs)


# =========================================================
# FUTURE PREDICTION (no retraining)
# =========================================================
def predict_future_congestion(model_path, block_graphs, tx_graphs, future_steps=10):
    """Predict next few blocks (1–10 steps) without retraining"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔮 Loading trained model from {model_path}...")
    
    # Load model
    model = HybridCongestionModel(
        block_in=block_graphs[0].x.shape[1],
        tx_in=tx_graphs[0].x.shape[1],
        hidden=64, out_dim=2, dropout=0.3
    ).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    preds, probs = [], []
    with torch.no_grad():
        for i in range(min(future_steps, len(block_graphs))):
            bb = Batch.from_data_list([block_graphs[-(i+1)]]).to(device)
            tt = Batch.from_data_list([tx_graphs[-(i+1)]]).to(device)
            out = model(bb, tt)
            prob = torch.softmax(out, dim=1)[:, 1].cpu().item()
            pred = out.argmax(dim=1).cpu().item()
            preds.append(pred)
            probs.append(prob)
    
    preds, probs = preds[::-1], probs[::-1]
    print(f"✅ Future {future_steps}-step Predictions: {preds}")
    print(f"   Probabilities: {[round(p, 3) for p in probs]}")
    return np.array(preds), np.array(probs)


# =========================================================
# MODEL CHECKPOINT SAVE/LOAD
# =========================================================
def save_model_checkpoint(model, metrics, filename="bitcoin_congestion_model.pt"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    os.makedirs("saved_models", exist_ok=True)
    torch.save(checkpoint, f"saved_models/{filename}")
    print(f"✅ Model checkpoint saved to saved_models/{filename}")


def load_model_checkpoint(model, filename="saved_models/bitcoin_congestion_model.pt"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded from {filename}")
    print(f"Metrics: {checkpoint['metrics']}")
    return model


# =========================================================
# EMBEDDING SAVE FUNCTION
# =========================================================
def save_validation_embeddings(model, val_blocks, val_txs, y_val, device="cpu"):
    """Save embeddings and labels for reuse"""
    os.makedirs("saved_outputs", exist_ok=True)
    model.eval()
    model = model.to(device)
    embeddings, labels = [], []
    with torch.no_grad():
        for b, t, y in zip(val_blocks, val_txs, y_val):
            bb = Batch.from_data_list([b]).to(device)
            tt = Batch.from_data_list([t]).to(device)
            block_emb = model.block_encoder(bb.x, bb.edge_index, bb.batch)
            tx_emb = model.tx_encoder(tt.x, tt.edge_index, tt.batch)
            emb = torch.cat([block_emb, tx_emb], dim=1)
            embeddings.append(emb.cpu().numpy())
            labels.append([y])
    np.save("saved_outputs/val_embeddings.npy", np.concatenate(embeddings))
    np.save("saved_outputs/y_val_true.npy", np.array(labels))
    print("✅ Validation embeddings and labels saved!")


# =========================================================
# MAIN BITCOIN WORKFLOW
# =========================================================
def main_bitcoin():
    """Main training pipeline for Bitcoin congestion"""
    print("=" * 70)
    print("BITCOIN BLOCKCHAIN CONGESTION PREDICTION")
    print("=" * 70)

    # 1) Load Bitcoin transactions
    print("\n[1] Loading Bitcoin transactions from MongoDB...")
    df = load_btc_transactions(
        mongo_uri="mongodb://localhost:27017/",
        db_name="btc_transactions_db",
        collection_name="decoded_transactions"
    )
    if df.empty:
        print("❌ Failed to load Bitcoin transactions. Exiting.")
        return
    print(f"Loaded {len(df)} transactions from {df['slot'].nunique()} blocks")

    # 2) Block stats + graph building
    print("\n[2] Building block-level statistics...")
    block_stats = build_block_stats_btc(df)
    print(f"Block stats shape: {block_stats.shape}")

    print("\n[3] Building block & transaction graphs...")
    block_graphs, labels, target_slots = build_block_graphs_btc(
        block_stats, window=8, congestion_metric='composite', threshold=None
    )
    tx_graphs = build_tx_graphs_btc(df, target_slots)
    print(f"Built {len(block_graphs)} block graphs and {len(tx_graphs)} tx graphs")

    assert len(block_graphs) == len(tx_graphs) == len(labels), "❌ Data mismatch!"

    # 4) Model initialization
    print("\n[4] Initializing model...")
    model = HybridCongestionModel(
        block_in=block_graphs[0].x.shape[1],
        tx_in=tx_graphs[0].x.shape[1],
        hidden=64, out_dim=2, dropout=0.3
    )

    # 5) Training
    print("\n[5] Training model...")
    trained_model = run_training(
        block_graphs, tx_graphs, labels, model, epochs=30, batch_size=16, lr=0.001
    )

    # 6) Evaluation & metrics
    print("\n[6] Evaluating model performance...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, probs = analyze_predictions(trained_model, block_graphs, tx_graphs, labels, device)
    y_true = labels.numpy() if torch.is_tensor(labels) else labels
    metrics = calculate_metrics(y_true, preds)
    print("Metrics:", metrics)

    # 7) Save embeddings + model
    print("\n[7] Saving embeddings and model...")
    save_validation_embeddings(trained_model, block_graphs, tx_graphs, labels)
    save_model_checkpoint(trained_model, metrics)

    print("\n✅ Training complete. Ready for inference.")
    return trained_model, block_graphs, tx_graphs, labels, block_stats, metrics


# =========================================================
# MAIN FULL WORKFLOW (with future prediction)
# =========================================================
def main_bitcoin_full():
    """Complete workflow including future forecasting"""
    model, block_graphs, tx_graphs, labels, block_stats, metrics = main_bitcoin()

    import pickle

    def save_data(data, filename='data_btc.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    # Save the data
    save_data((block_graphs, tx_graphs, labels, block_stats))


    # Predict current performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds, probs = analyze_predictions(model, block_graphs, tx_graphs, labels, device)

    # Target slots
    target_slots = [g.slot for g in block_graphs]

    # Save summary & CSV export
    #export_df = export_predictions_to_csv(block_stats, preds, probs, target_slots)

    # Predict next 10 future congestion states (no retrain)
    print("\n[8] Predicting future 10-step congestion...")
    future_preds, future_probs = predict_future_congestion(
        "saved_models/bitcoin_congestion_model.pt", block_graphs, tx_graphs, future_steps=10
    )

    print("\n✅ Pipeline Complete! Future Predictions Generated.")
    return model, block_stats, metrics


# =========================================================
# MAIN ENTRY
# =========================================================
if __name__ == "__main__":
    model, block_stats, metrics = main_bitcoin_full()


# In[ ]:




