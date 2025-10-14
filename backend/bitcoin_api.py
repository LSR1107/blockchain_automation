import requests
from datetime import datetime

BITCOIN_RPC = "https://bitcoin-mainnet.g.alchemy.com/v2/lxecqlqSRDkowA0inWLcrEt7GsnrT9Fs"

# cache so we don’t fetch same txid multiple times
UTXO_CACHE = {}

def rpc_call(method, params=[]):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(BITCOIN_RPC, json=payload)
    r.raise_for_status()
    return r.json().get("result")

def get_block_hash(height):
    return rpc_call("getblockhash", [height])

def get_block(block_hash, verbose=True):
    # verbose=2 → includes tx details with vin/vout
    return rpc_call("getblock", [block_hash, 2 if verbose else 1])

def get_raw_transaction(txid):
    if txid in UTXO_CACHE:
        return UTXO_CACHE[txid]
    tx = rpc_call("getrawtransaction", [txid, True])
    UTXO_CACHE[txid] = tx
    return tx

def calculate_tx_fee(tx):
    # Skip coinbase (no vin)
    if "vin" not in tx or not tx["vin"] or "coinbase" in tx["vin"][0]:
        return 0

    input_sum = 0
    for vin in tx["vin"]:
        if "txid" in vin and "vout" in vin:
            prev_tx = get_raw_transaction(vin["txid"])
            prev_out = prev_tx["vout"][vin["vout"]]
            input_sum += prev_out.get("value", 0)

    output_sum = sum(vout.get("value", 0) for vout in tx.get("vout", []))
    return max(input_sum - output_sum, 0)

def get_block_with_fees(block_hash):
    block = get_block(block_hash)
    if not block:
        return None, 0

    total_fees = 0
    for tx in block.get("tx", []):
        total_fees += calculate_tx_fee(tx)

    return block, total_fees

def get_latest_block_height():
    return rpc_call("getblockcount")

def get_latest_btc_block_info():
    latest_height = get_latest_block_height()
    block_hash = get_block_hash(latest_height)
    block, total_fees = get_block_with_fees(block_hash)

    txs = block.get("tx", [])
    block_time = datetime.utcfromtimestamp(block["time"]).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "height": latest_height,
        "hash": block_hash,
        "block_time": block_time,
        "transaction_count": len(txs),
        "total_fees": total_fees,
        "avg_fee_per_tx": total_fees / len(txs) if txs else 0,
    }
def get_recent_btc_blocks(n_blocks=10):
    latest_height = get_latest_block_height()
    blocks = []
    for height in range(latest_height, latest_height - n_blocks, -1):
        block_hash = get_block_hash(height)
        block, total_fees = get_block_with_fees(block_hash)
        if not block:
            continue
        txs = block.get("tx", [])
        block_time = datetime.utcfromtimestamp(block["time"]).strftime("%Y-%m-%d %H:%M:%S")
        blocks.append({
            "height": height,
            "hash": block_hash,
            "timestamp": block_time,
            "tx_count": len(txs),
            "total_fees": total_fees,
            "avg_fee_per_tx": total_fees / len(txs) if txs else 0
        })
    return blocks
