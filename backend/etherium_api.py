import requests
from datetime import datetime

# Replace with your Ethereum Alchemy RPC URL
RPC_URL = "https://eth-mainnet.g.alchemy.com/v2/lxecqlqSRDkowA0inWLcrEt7GsnrT9Fs"


def rpc_call(method, params=None, rpc_url=RPC_URL):
    """Helper to make RPC calls."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or []}
    res = requests.post(rpc_url, json=payload).json()
    return res.get("result")


def get_latest_block_number(rpc_url=RPC_URL):
    """Get latest Ethereum block number (int)."""
    block_hex = rpc_call("eth_blockNumber", [], rpc_url)
    return int(block_hex, 16) if block_hex else None


def get_block(block_number, rpc_url=RPC_URL):
    """Fetch full block data by block number."""
    block_hex = hex(block_number)
    return rpc_call("eth_getBlockByNumber", [block_hex, True], rpc_url)


def get_latest_eth_block_info():
    """Get latest block info: number, timestamp, tx count, gas usage, fees, TPS est."""
    latest_block_num = get_latest_block_number()
    block_data = get_block(latest_block_num)

    if not block_data:
        return None

    # Block time
    timestamp = int(block_data.get("timestamp", "0x0"), 16)
    block_time_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

    # Transactions
    txs = block_data.get("transactions", [])
    tx_count = len(txs)

    # Gas used
    gas_used = int(block_data.get("gasUsed", "0x0"), 16)
    gas_limit = int(block_data.get("gasLimit", "0x0"), 16)

    # Approx total fees (sum gasUsed * gasPrice)
    total_fees = 0
    for tx in txs:
        try:
            gas_price = int(tx.get("gasPrice", "0x0"), 16)
            gas = int(tx.get("gas", "0x0"), 16)
            total_fees += gas * gas_price
        except Exception:
            continue

    return {
        "block_number": latest_block_num,
        "block_time": block_time_str,
        "transaction_count": tx_count,
        "gas_used": gas_used,
        "gas_limit": gas_limit,
        "gas_utilization": (gas_used / gas_limit * 100) if gas_limit else 0,
        "total_fees_eth": total_fees / 1e18,  # in ETH
        "avg_fee_per_tx_eth": (total_fees / 1e18 / tx_count) if tx_count else 0,
        # Ethereum ~12s per block → TPS estimate
        "tps_estimate": tx_count / 12 if tx_count else 0,
    }


def get_recent_eth_blocks(n_blocks=50):
    """Fetch last n blocks and compute metrics."""
    latest_block_num = get_latest_block_number()
    blocks = []

    for num in range(latest_block_num, latest_block_num - n_blocks, -1):
        block = get_block(num)
        if not block:
            continue

        timestamp = int(block.get("timestamp", "0x0"), 16)
        block_time_str = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        txs = block.get("transactions", [])
        tx_count = len(txs)

        gas_used = int(block.get("gasUsed", "0x0"), 16)
        gas_limit = int(block.get("gasLimit", "0x0"), 16)

        total_fees = 0
        for tx in txs:
            try:
                gas_price = int(tx.get("gasPrice", "0x0"), 16)
                gas = int(tx.get("gas", "0x0"), 16)
                total_fees += gas * gas_price
            except Exception:
                continue

        blocks.append({
            "block_number": num,
            "timestamp": block_time_str,
            "tx_count": tx_count,
            "gas_used": gas_used,
            "gas_limit": gas_limit,
            "gas_utilization": (gas_used / gas_limit * 100) if gas_limit else 0,
            "total_fees_eth": total_fees / 1e18,
            "avg_fee_per_tx_eth": (total_fees / 1e18 / tx_count) if tx_count else 0,
            "tps_estimate": tx_count / 12 if tx_count else 0
        })

    return blocks
