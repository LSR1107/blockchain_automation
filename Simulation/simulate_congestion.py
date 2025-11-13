import json
import time
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.message import Message
from solders.transaction import Transaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solana.rpc.api import Client
from solders.signature import Signature
from typing import cast

# --- Configuration ---
WALLET_FILE = r"C:\Users\lohit\.config\solana\devnet.json"
RPC_URL = "https://api.devnet.solana.com"
RECIPIENT_PUBKEY = Pubkey.from_string("6Mzk8SNy86m8U7NxhcJ4C3GAwFjwNNV4viXEY622JmBt")
TRANSFER_AMOUNT_SOL = 0.001  # 0.001 SOL for testing

# --- Load wallet ---
with open(WALLET_FILE, "r") as f:
    secret = json.load(f)
wallet = Keypair.from_bytes(bytes(secret))
client = Client(RPC_URL, timeout=30)

# --- Maps for congestion ---
priority_fee_map = {"low": 5, "medium": 50, "high": 150}          # μLamports / CU
compute_unit_limit_map = {"low": 20000, "medium": 40000, "high": 80000}


def simulate_transaction(congestion="medium"):
    """Simulate a transaction on Solana Devnet with congestion-based priority fees."""
    priority_fee = priority_fee_map.get(congestion, 50)
    compute_limit = compute_unit_limit_map.get(congestion, 40000)

    # Build instructions
    instructions = [
        set_compute_unit_price(priority_fee),
        set_compute_unit_limit(compute_limit),
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(TRANSFER_AMOUNT_SOL * 1e9),
            )
        ),
    ]

    try:
        recent_blockhash = client.get_latest_blockhash().value.blockhash
        message = Message.new_with_blockhash(instructions, wallet.pubkey(), recent_blockhash)
        txn = Transaction.new_unsigned(message)
        txn.sign([wallet], recent_blockhash)

        # Send and confirm
        start_time = time.time()
        response = client.send_transaction(txn)
        signature_obj = cast(Signature, response.value)
        signature_str = str(signature_obj)
        confirmed = client.confirm_transaction(signature_obj, commitment="confirmed")
        end_time = time.time()

        confirmation_time = round(end_time - start_time, 3)
        status = "Confirmed" if confirmed.value[0].err is None else "Failed"

        # Fetch transaction fee
        time.sleep(1)
        tx_info = client.get_transaction(signature_obj, encoding="jsonParsed", commitment="confirmed")
        actual_fee = None
        if tx_info.value and tx_info.value.transaction.meta:
            actual_fee = tx_info.value.transaction.meta.fee / 1e9

        result = {
            "congestion": congestion,
            "priority_fee": priority_fee,
            "compute_limit": compute_limit,
            "status": status,
            "confirmation_time": confirmation_time,
            "fee_SOL": round(actual_fee or 0.0, 9),
            "signature": signature_str,
        }
        return result

    except Exception as e:
        return {"error": str(e)}
