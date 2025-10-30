#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


def load_eth_transactions(
    mongo_uri="mongodb://localhost:27017/",
    db_name="eth_transactions_db",
    collection_name="decoded_transactions"
):
    """
    Load Ethereum transactions from MongoDB with enhanced feature extraction.
    Compatible with Solana pipeline structure.
    Handles missing gas_used and gas_limit gracefully.
    """
    from pymongo import MongoClient
    import pandas as pd
    import numpy as np

    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Pull documents with all relevant fields
    docs = list(collection.find({}, {
        "block_number": 1,
        "block_time": 1,
        "tx_hash": 1,
        "from": 1,
        "to": 1,
        "gas_price": 1,
        "gas_used": 1,
        "gas_limit": 1,
        "max_fee_per_gas": 1,
        "max_priority_fee_per_gas": 1,
        "nonce": 1,
        "type": 1,
        "status": 1,
        "function": 1,
        "value": 1
    }))

    df = pd.DataFrame(docs)

    if df.empty:
        print("⚠️ No Ethereum transactions found in MongoDB.")
        return df

    # Rename for consistency with Solana schema
    df.rename(columns={
        "block_number": "slot",
        "block_time": "blockTime",
        "tx_hash": "signature"
    }, inplace=True)

    # Convert to datetime
    if "blockTime" in df.columns:
        df["blockTime"] = pd.to_datetime(df["blockTime"], errors="coerce")

    # Ensure all numeric fields exist and are clean
    numeric_cols = [
        "gas_price", "gas_used", "gas_limit",
        "max_fee_per_gas", "max_priority_fee_per_gas",
        "nonce", "value"
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Fill missing gas_used and gas_limit with reasonable defaults
    # since they don't exist in MongoDB data
    df["gas_used"] = np.where(df["gas_used"] == 0, 21_000, df["gas_used"])        # avg per tx
    df["gas_limit"] = np.where(df["gas_limit"] == 0, 30_000_000, df["gas_limit"]) # block-level limit

    # Derived gas efficiency (safety with epsilon)
    df["gas_efficiency"] = df["gas_used"] / (df["gas_limit"] + 1e-9)

    # Effective gas price (EIP-1559 handling)
    df["effective_gas_price"] = df[["gas_price", "max_fee_per_gas"]].max(axis=1)

    # Instruction complexity proxy
    df["instructionCount"] = df["function"].apply(lambda x: len(str(x)) if pd.notna(x) else 1)

    # Optional: verified tx indicator
    df["is_verified"] = df["status"].apply(lambda s: 1 if str(s).lower() == "verified" else 0)

    # Sort chronologically for later graph windows
    df = df.sort_values("blockTime").reset_index(drop=True)

    print(f"✅ Loaded {len(df)} Ethereum transactions across {df['slot'].nunique()} blocks.")
    print(f"Sample columns: {list(df.columns)}")
    return df


# In[4]:


import numpy as np
import pandas as pd

def build_block_stats_eth(transactions_df):
    """
    Build block-level statistics for Ethereum, compatible with your Mongo-extracted data
    where columns are: slot, blockTime, gas_price, gas_used, gas_limit, value, etc.
    """

    if transactions_df.empty:
        return pd.DataFrame()

    df = transactions_df.copy()

    # ✅ Rename columns to expected lowercase versions
    df.rename(columns={
        "blockTime": "timestamp",
        "gas_price": "gasPrice",
        "gas_used": "gasUsed",
        "gas_limit": "gasLimit",
    }, inplace=True)

    print(f"\nProcessing {len(df)} transactions from {df['slot'].nunique()} blocks...")

    # Group by slot (equivalent to block number)
    block_groups = df.groupby("slot")

    block_stats = []

    for slot, group in block_groups:
        tx_count = len(group)
        success_count = len(group[group.get("status", 1) == 1]) if "status" in group.columns else tx_count
        fail_count = tx_count - success_count

        total_gas_used = group["gasUsed"].sum() if "gasUsed" in group.columns else 0
        gas_limit = group["gasLimit"].iloc[0] if "gasLimit" in group.columns else 30_000_000

        block_utilization = total_gas_used / gas_limit if gas_limit > 0 else 0

        if "gasPrice" in group.columns:
            gas_prices_gwei = group["gasPrice"] / 1e9
            avg_gas_price = gas_prices_gwei.mean()
            median_gas_price = gas_prices_gwei.median()
            fee_variance = gas_prices_gwei.std()
        else:
            avg_gas_price = median_gas_price = fee_variance = 0

        if "value" in group.columns:
            values_eth = group["value"] / 1e18
            avg_value = values_eth.mean()
            median_value = values_eth.median()
            total_value = values_eth.sum()
        else:
            avg_value = median_value = total_value = 0

        timestamp = group["timestamp"].iloc[0] if "timestamp" in group.columns else np.nan

        if "gasUsed" in group.columns and "gas_limit" in group.columns:
            gas_requested = group["gas_limit"].sum()
            avg_gas_efficiency = (total_gas_used / gas_requested) if gas_requested > 0 else 0
        else:
            avg_gas_efficiency = 0

        base_fee = group["max_fee_per_gas"].iloc[0] / 1e9 if "max_fee_per_gas" in group.columns else 0

        block_stats.append({
            "slot": slot,
            "block_number": slot,
            "timestamp": timestamp,
            "tx_count": tx_count,
            "success_count": success_count,
            "fail_count": fail_count,
            "total_gas_used": total_gas_used,
            "gas_limit": gas_limit,
            "block_utilization": block_utilization,
            "avg_gas_price": avg_gas_price,
            "median_gas_price": median_gas_price,
            "fee_variance": fee_variance,
            "base_fee_per_gas": base_fee,
            "avg_value": avg_value,
            "median_value": median_value,
            "total_value": total_value,
            "avg_gas_efficiency": avg_gas_efficiency,
        })

    bs = pd.DataFrame(block_stats).sort_values("slot").reset_index(drop=True)

    # TPS computation
    bs["block_time_delta"] = bs["timestamp"].diff().dt.total_seconds().fillna(12)
    bs["tps"] = bs["tx_count"] / bs["block_time_delta"]
    bs["tps"] = bs["tps"].replace([np.inf, -np.inf], 0).fillna(0)

    # Normalization for congestion score
    bs["util_norm"] = bs["block_utilization"].clip(0, 1)
    bs["tps_norm"] = (bs["tps"] - bs["tps"].min()) / (bs["tps"].max() - bs["tps"].min() + 1e-8)
    bs["fee_norm"] = (bs["avg_gas_price"] - bs["avg_gas_price"].min()) / (
        bs["avg_gas_price"].max() - bs["avg_gas_price"].min() + 1e-8
    )

    bs["congestion_score"] = (
        0.4 * bs["util_norm"] +
        0.3 * bs["tps_norm"] +
        0.3 * bs["fee_norm"]
    )

    print("\n✅ Ethereum Block Statistics Built Successfully ✅")
    print(f"Total Blocks: {len(bs)}, Total Transactions: {bs['tx_count'].sum():,}")
    print(f"Avg Utilization: {bs['block_utilization'].mean():.4f}, Avg TPS: {bs['tps'].mean():.2f}")

    return bs


# In[5]:


def build_block_graphs(block_stats, window=16, congestion_metric='composite', threshold=None):
    """
    Build block-level graphs with multiple congestion metrics
    
    Args:
        block_stats: DataFrame from build_block_stats()
        window: Sliding window size
        congestion_metric: 'utilization', 'tps', 'composite', or 'multi'
        threshold: Congestion threshold (auto-determined if None)
    """
    bs = block_stats.sort_values("slot").reset_index(drop=True).copy()
    
    # Select feature columns - NOW INCLUDES TPS!
    feat_cols = [
        "tx_count", "success_count", "fail_count",
        "avg_gas_price", "median_gas_price", "fee_variance",
        "block_time_delta", "block_utilization",
        "tps",  # 🔥 TPS added!
        "congestion_score"  # 🔥 Composite score added!
    ]
    
    # Add optional features
    optional_cols = ["avg_value", "median_value", "total_gas_used", "avg_gas_efficiency", "base_fee_per_gas"]
    for col in optional_cols:
        if col in bs.columns:
            feat_cols.append(col)
    
    X = bs[feat_cols].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    # Determine threshold automatically if not provided
    if threshold is None:
        if congestion_metric == 'utilization':
            # Use 60th percentile for utilization
            threshold = bs['block_utilization'].quantile(0.60)
            metric_col = 'block_utilization'
        elif congestion_metric == 'tps':
            # Use 60th percentile for TPS
            threshold = bs['tps'].quantile(0.60)
            metric_col = 'tps'
        elif congestion_metric == 'composite':
            # Use 75th percentile for composite score
            threshold = bs['congestion_score'].quantile(0.75)
            metric_col = 'congestion_score'
        else:  # multi
            # Use multiple criteria
            util_thresh = bs['block_utilization'].quantile(0.70)
            tps_thresh = bs['tps'].quantile(0.70)
            metric_col = None
    else:
        if congestion_metric == 'utilization':
            metric_col = 'block_utilization'
        elif congestion_metric == 'tps':
            metric_col = 'tps'
        else:
            metric_col = 'congestion_score'
    
    print(f"\n{'='*70}")
    print(f"Building graphs with congestion_metric='{congestion_metric}'")
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
        
        # 🔥 Create label based on chosen metric
        if congestion_metric == 'multi':
            # Multi-criteria: congested if EITHER utilization OR tps is high
            y = int((bs.loc[i, "block_utilization"] > util_thresh) or 
                   (bs.loc[i, "tps"] > tps_thresh))
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


def build_tx_graphs_eth(df, target_slots):
    """
    Build transaction-level graphs for each target slot
    """
    # Prepare features
    feat_cols = ["effective_gas_price", "value", "instructionCount", "gas_efficiency"]
    available_cols = [col for col in feat_cols if col in df.columns]
    
    X = df[available_cols].fillna(0).to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    # Group by slot
    slot2idxs = df.groupby("slot").indices
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
                # Create edges based on transaction order
                src = list(range(n - 1))
                dst = list(range(1, n))
                edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data.slot = int(s)
        graphs.append(data)
    
    return graphs


# In[7]:



# =================== MODEL COMPONENTS ===================
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

# =================== DATASET HANDLING ===================
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
    train_blocks = [block_graphs[i] for i in idx_train]
    test_blocks = [block_graphs[i] for i in idx_test]
    train_txs = [tx_graphs[i] for i in idx_train]
    test_txs = [tx_graphs[i] for i in idx_test]
    y_train = labels[list(idx_train)]
    y_test = labels[list(idx_test)]
    return train_blocks, test_blocks, train_txs, test_txs, y_train, y_test

# =================== TRAINING & EVALUATION ===================
def train(model, loader, optimizer, criterion, device):
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
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for block_data, tx_data, labels in loader:
            block_data, tx_data, labels = block_data.to(device), tx_data.to(device), labels.to(device)
            out = model(block_data, tx_data)
            loss = criterion(out, labels)
            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), correct / len(loader.dataset), np.array(preds_all), np.array(labels_all)

# =================== TRAINING RUNNER ===================
def run_training(block_graphs, tx_graphs, labels, model, epochs=30, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_blocks, test_blocks, train_txs, test_txs, y_train, y_test = split_dataset(block_graphs, tx_graphs, labels)
    
    train_dataset = HybridDataset(train_blocks, train_txs, y_train)
    test_dataset = HybridDataset(test_blocks, test_txs, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, preds, y_true = evaluate(model, test_loader, criterion, device)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_hybrid_model.pth")

        print(f"Epoch {epoch:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {tr_acc:.3f} | Val Acc: {val_acc:.3f}")

    # Plot metrics
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Validation Acc')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.show()

    print(f"✅ Best Validation Accuracy: {best_acc:.4f}")

    # Extra metrics
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    print(f"\nMAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

    print("\n[Saving validation embeddings and true labels...]")

    model.eval()
    embeddings, y_val_true = [], []

    with torch.no_grad():
        for block_data, tx_data, y_val in test_loader:
            block_data, tx_data, y_val = block_data.to(device), tx_data.to(device), y_val.to(device)
            
            # Forward pass
            block_emb = model.block_encoder(block_data.x, block_data.edge_index, block_data.batch)
            tx_emb = model.tx_encoder(tx_data.x, tx_data.edge_index, tx_data.batch)
            combined_emb = torch.cat([block_emb, tx_emb], dim=1)
            
            embeddings.append(combined_emb.cpu().numpy())
            y_val_true.append(y_val.cpu().numpy())

    GNN_val_embeddings = np.concatenate(embeddings, axis=0)
    GNN_y_val_true = np.concatenate(y_val_true, axis=0)

    #os.makedirs("saved_outputs", exist_ok=True)
    np.save("GNN_val_embeddings.npy", GNN_val_embeddings)
    np.save("GNN_y_val_true.npy", GNN_y_val_true)
    
    #print(f"✅ Saved val embeddings: {embeddings.shape}, y_val: {y_val_true.shape}")
    print("Embeddings and true labels stored in 'saved_outputs/'")
    
    return model

# =================== FUTURE PREDICTIONS ===================
def predict_future(model, future_data, device, steps=10):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(steps):
            block_data, tx_data = future_data[i]
            block_data, tx_data = block_data.to(device), tx_data.to(device)
            out = model(block_data, tx_data)
            preds.append(out)
    return np.array(preds)


# In[8]:


# =================== MAIN EXECUTION ===================
def main_ethereum():
    """
    Main function for Ethereum blockchain congestion prediction
    with full training, evaluation, and short-term future forecasting.
    """
    print("=" * 60)
    print("ETHEREUM BLOCKCHAIN CONGESTION PREDICTION")
    print("=" * 60)

    # 1) Load Ethereum transactions
    print("\n[1] Loading Ethereum transactions from MongoDB...")
    df = load_eth_transactions(
        mongo_uri="mongodb://localhost:27017/",
        db_name="eth_transactions_db",
        collection_name="decoded_transactions"
    )
    print(f"✅ Loaded {len(df)} transactions from {df['slot'].nunique()} blocks")

    # 2) Build block-level statistics
    print("\n[2] Building block-level statistics...")
    block_stats = build_block_stats_eth(df)
    print(f"✅ Block stats shape: {block_stats.shape}")
    print(f"📊 Average block utilization: {block_stats['block_utilization'].mean():.3f}")

    # 3) Build block graphs
    print("\n[3] Building block graphs...")
    median_util = block_stats['block_utilization'].median()
    print(f"ℹ️ Using median threshold: {median_util:.4f}")
    block_graphs, labels, target_slots = build_block_graphs(
        block_stats,
        window=8,
        congestion_metric='composite',
        threshold=None
    )
    print(f"✅ Created {len(block_graphs)} block graphs")

    # 4) Build transaction graphs
    print("\n[4] Building transaction graphs...")
    tx_graphs = build_tx_graphs_eth(df, target_slots)
    print(f"✅ Created {len(tx_graphs)} transaction graphs")

    # Sanity check
    assert len(block_graphs) == len(tx_graphs), "Mismatch in graph counts!"
    assert len(labels) == len(block_graphs), "Mismatch in label count!"

    # 5) Initialize model
    print("\n[5] Initializing hybrid GNN model...")
    block_in_dim = block_graphs[0].x.shape[1] if block_graphs else 8
    tx_in_dim = tx_graphs[0].x.shape[1] if tx_graphs else 4
    print(f"📐 Block input dim: {block_in_dim}, Tx input dim: {tx_in_dim}")

    model = HybridCongestionModel(
        block_in=block_in_dim,
        tx_in=tx_in_dim,
        hidden=64,
        out_dim=2,
        dropout=0.3
    )

    # 6) Train and validate model
    print("\n[6] Starting training and validation...")
    trained_model = run_training(
        block_graphs,
        tx_graphs,
        labels,
        model,
        epochs=30,
        batch_size=4,
        lr=0.001
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

    # 7) Load best model weights for inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.load_state_dict(torch.load("best_hybrid_model.pth", map_location=device))
    trained_model = trained_model.to(device)
    trained_model.eval()
    

    # 8) Future predictions (1–10 next time steps)
    print("\n[7] Running short-term future predictions (next 1–10 steps)...")
    # Prepare a few unseen future graphs — e.g., last 10 graphs from the dataset
    future_data = list(zip(block_graphs[-10:], tx_graphs[-10:]))

    future_preds = predict_future(trained_model, future_data, device, steps=10)
    print("✅ Future Predictions (Class IDs):", future_preds.flatten().tolist())

    print("\n[8] Pipeline execution complete.\n")
    print("=" * 60)
    print("🚀 ETHEREUM HYBRID GNN FORECAST PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 60)

    return trained_model, block_graphs, tx_graphs, labels


# In[9]:


# =================== ADDITIONAL ANALYSIS FUNCTIONS ===================
def analyze_predictions(model, block_graphs, tx_graphs, labels, device='cpu'):
    """
    Analyze model predictions for insights
    """
    model.eval()
    model = model.to(device)
    
    dataset = HybridDataset(block_graphs, tx_graphs, labels.numpy() if torch.is_tensor(labels) else labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for block_data, tx_data, _ in loader:
            block_data, tx_data = block_data.to(device), tx_data.to(device)
            out = model(block_data, tx_data)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of congestion
    
    return np.array(all_preds), np.array(all_probs)

def calculate_metrics(y_true, y_pred):
    """
    Calculate detailed performance metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm}


# In[11]:


# Run the main function
if __name__ == "__main__":
    # Train the model
    model, block_graphs, tx_graphs, labels = main_ethereum()
    
    import pickle

    def save_data(data, filename='data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    # Save the data
    save_data((block_graphs, tx_graphs, labels))


# Save the model and other variables
    save_model_and_data(model, block_graphs, tx_graphs, labels)

    
    # Analyze predictions
    print("\n[7] Analyzing model predictions...")
    preds, probs = analyze_predictions(model, block_graphs, tx_graphs, labels)
    
    # Calculate metrics
    print("\n[8] Performance Metrics:")
    metrics = calculate_metrics(labels.numpy() if torch.is_tensor(labels) else labels, preds)


# In[ ]:




