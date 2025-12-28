import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

BITCOIN_RPC = "https://bitcoin-mainnet.g.alchemy.com/v2/zjOZa00S_2kWiyXY2wIxE"
HEADERS = {"Content-Type": "application/json"}

# ================== RPC ==================

def rpc_call(method, params=[]):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(BITCOIN_RPC, json=payload, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("result")

def get_latest_block_height():
    return rpc_call("getblockcount")

def get_block_hash(height):
    return rpc_call("getblockhash", [height])

def get_block_time(block_hash):
    block = rpc_call("getblock", [block_hash, 1])
    return datetime.utcfromtimestamp(block["time"]).strftime("%Y-%m-%d %H:%M:%S")

def get_block_stats(block_hash):
    return rpc_call("getblockstats", [block_hash])

# ================== SINGLE BLOCK ==================

def analyze_block(height):
    try:
        block_hash = get_block_hash(height)
        stats = get_block_stats(block_hash)
        timestamp = get_block_time(block_hash)

        total_fees_btc = stats["totalfee"] / 1e8
        avg_fee = total_fees_btc / stats["txs"] if stats["txs"] else 0

        return {
            "height": height,
            "hash": block_hash,
            "timestamp": timestamp,
            "transaction_count": stats["txs"],
            "block_size_bytes": stats["total_size"],
            "input_count": stats["ins"],
            "output_count": stats["outs"],
            "total_fees_btc": total_fees_btc,
            "avg_fee_per_tx_btc": avg_fee,
        }
    except Exception as e:
        return {"height": height, "error": str(e)}

# ================== PARALLEL ==================

def _get_recent_btc_blocks(n=10):
    latest = get_latest_block_height()
    heights = list(range(latest, latest - n, -1))

    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(analyze_block, heights))

# ================== CACHE ==================

_CACHE = {"data": None, "time": 0}

def get_recent_btc_blocks(n_blocks=10, ttl=30):
    now = time.time()

    if _CACHE["data"] is None or now - _CACHE["time"] > ttl:
        _CACHE["data"] = _get_recent_btc_blocks(n_blocks)
        _CACHE["time"] = now

    return _CACHE["data"]

# ================== LATEST BLOCK ==================

def get_latest_btc_block_info():
    height = get_latest_block_height()
    return analyze_block(height)
