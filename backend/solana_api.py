import requests
from datetime import datetime

RPC_URL = "https://solana-mainnet.g.alchemy.com/v2/lxecqlqSRDkowA0inWLcrEt7GsnrT9Fs"


def get_latest_slot(rpc_url=RPC_URL):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "getSlot"}
    res = requests.post(rpc_url, json=payload).json()
    return res.get("result")


def get_block(slot, rpc_url=RPC_URL):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getBlock",
        "params": [
            slot,
            {
                "encoding": "json",
                "transactionDetails": "full",
                "rewards": False,
                "maxSupportedTransactionVersion": 0
            }
        ]
    }
    r = requests.post(rpc_url, json=payload).json()
    return r.get("result", None)


def get_latest_block_info():
    """Get latest block slot, timestamp, txn count, fees, TPS."""
    latest_slot = get_latest_slot()
    block_data = get_block(latest_slot)

    if not block_data:
        return None

    block_time = block_data.get("blockTime")
    block_time_str = (
        datetime.utcfromtimestamp(block_time).strftime("%Y-%m-%d %H:%M:%S")
        if block_time else "N/A"
    )

    txs = block_data.get("transactions", [])
    fees = sum(tx.get("meta", {}).get("fee", 0) for tx in txs)

    return {
        "slot": latest_slot,
        "block_time": block_time_str,
        "transaction_count": len(txs),
        "total_fees": fees,
        "avg_fee_per_tx": fees / len(txs) if txs else 0,
        "tps_estimate": len(txs) / 0.4 if txs else 0
    }


def get_recent_blocks(n_blocks=50):
    """Fetch last n blocks and compute metrics (tx count, fees, TPS)."""
    latest_slot = get_latest_slot()
    blocks = []
    for slot in range(latest_slot, latest_slot - n_blocks, -1):
        block = get_block(slot)
        if not block:
            continue
        block_time = block.get("blockTime")
        txs = block.get("transactions", [])
        fees = sum(tx.get("meta", {}).get("fee", 0) for tx in txs)
        blocks.append({
            "slot": slot,
            "timestamp": datetime.utcfromtimestamp(block_time).strftime("%Y-%m-%d %H:%M:%S") if block_time else "N/A",
            "tx_count": len(txs),
            "total_fees": fees,
            "avg_fee_per_tx": fees / len(txs) if txs else 0,
            "tps_estimate": len(txs) / 0.4 if txs else 0
        })
    return blocks
