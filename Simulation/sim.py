import json
import time
import random
import csv
from pathlib import Path
from typing import cast
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.message import Message
from solders.transaction import Transaction
from solders.compute_budget import set_compute_unit_limit, set_compute_unit_price
from solana.rpc.api import Client

# --- Configuration ---
WALLET_FILE = r"C:\Users\lohit\.config\solana\devnet.json"
RPC_URL = "https://api.devnet.solana.com"
RECIPIENT_PUBKEY = Pubkey.from_string("6Mzk8SNy86m8U7NxhcJ4C3GAwFjwNNV4viXEY622JmBt")
NUM_TRANSACTIONS = 10
OUTPUT_FILE = Path("solana_simulation_results.csv")

# --- Load wallet keypair ---
with open(WALLET_FILE, "r") as f:
    secret = json.load(f)
wallet = Keypair.from_bytes(bytes(secret))

# --- Solana client ---
client = Client(RPC_URL, timeout=30)

# ---------------- Transaction Types ----------------
def simulate_basic_transfer():
    return [
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.01 * 1e9),
            )
        )
    ], "Basic Transfer"

def simulate_priority_fee_transfer():
    priority_fee = random.randint(1, 100)  # μlamports per CU
    return [
        set_compute_unit_price(priority_fee),
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.01 * 1e9),
            )
        )
    ], f"Priority Fee Transfer ({priority_fee} μL/CU)"

def simulate_compute_limit_transfer():
    compute_units = random.randint(10_000, 50_000)
    return [
        set_compute_unit_limit(compute_units),
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.01 * 1e9),
            )
        )
    ], f"Compute Limited Transfer ({compute_units} CU)"

def simulate_multi_instruction_transaction():
    num_transfers = random.randint(2, 5)
    instructions = [
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.001 * 1e9),
            )
        )
        for _ in range(num_transfers)
    ]
    return instructions, f"Multi-Instruction Transaction ({num_transfers} transfers)"

def simulate_priority_multi_transaction():
    priority_fee = random.randint(50, 200)
    compute_units = random.randint(20_000, 100_000)
    return [
        set_compute_unit_price(priority_fee),
        set_compute_unit_limit(compute_units),
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.005 * 1e9),
            )
        ),
        transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(0.005 * 1e9),
            )
        )
    ], f"Complex Transaction ({priority_fee} μL/CU, {compute_units} CU, 2 transfers)"

TRANSACTION_TYPES = [
    simulate_basic_transfer,
    simulate_priority_fee_transfer,
    simulate_compute_limit_transfer,
    simulate_multi_instruction_transaction,
    simulate_priority_multi_transaction,
]

# ---------------- Transaction Sender ----------------
def send_transaction(instructions, description, tx_num):
    try:
        # Blockhash
        recent_blockhash = client.get_latest_blockhash().value.blockhash

        # Build + sign
        message = Message.new_with_blockhash(instructions, wallet.pubkey(), recent_blockhash)
        txn = Transaction.new_unsigned(message)
        txn.sign([wallet], recent_blockhash)

        # Send
        start_time = time.time()
        response = client.send_transaction(txn)
        signature = cast(str, response.value)

        # Confirm
        confirmed = client.confirm_transaction(signature, commitment="confirmed")
        end_time = time.time()
        status = "Confirmed" if confirmed.value[0].err is None else "Failed"
        confirmation_time = end_time - start_time

        # Get extra info
        time.sleep(1)
        tx_info = client.get_transaction(signature, encoding="jsonParsed", commitment="confirmed")
        fee, slot, block_time = None, None, None

        if tx_info.value:
            tx_dict = json.loads(tx_info.value.to_json())
            if "meta" in tx_dict and tx_dict["meta"]:
                fee = tx_dict["meta"]["fee"] / 1e9
            slot = tx_dict.get("slot")
            block_time = tx_dict.get("blockTime")

        print(f"Transaction {tx_num}:")
        print(f"  Signature: {signature}")
        print(f"  Status: {status}")
        print(f"  Time to confirm: {confirmation_time:.3f} sec")
        print(f"  Fee: {fee if fee else 'N/A'} SOL")
        print(f"  Description: {description}")
        print()

        return {
            "tx_no": tx_num,
            "signature": signature,
            "status": status,
            "confirmation_time": confirmation_time,
            "fee": fee,
            "slot": slot,
            "block_time": block_time,
            "type": description,
        }

    except Exception as e:
        print(f"  Transaction {tx_num} failed: {e}\n")
        return None

# ---------------- Main Simulation ----------------
def main():
    print("Solana Enhanced Simulation")
    print("=" * 60)

    initial_balance = client.get_balance(wallet.pubkey()).value / 1e9
    print(f"Initial Balance: {initial_balance:.6f} SOL\n")

    results = []
    for i in range(1, NUM_TRANSACTIONS + 1):
        tx_type = random.choice(TRANSACTION_TYPES)
        instructions, description = tx_type()
        result = send_transaction(instructions, description, i)
        if result:
            results.append(result)
        time.sleep(2)

    # Save CSV
    # with open(OUTPUT_FILE, "w", newline="") as f:
    #     writer = csv.DictWriter(
    #         f,
    #         fieldnames=[
    #             "tx_no",
    #             "signature",
    #             "status",
    #             "confirmation_time",
    #             "fee",
    #             "slot",
    #             "block_time",
    #             "type",
    #         ],
    #     )
    #     writer.writeheader()
    #     writer.writerows(results)

    final_balance = client.get_balance(wallet.pubkey()).value / 1e9
    print("=" * 60)
    print(f"Simulation Complete! {len(results)}/{NUM_TRANSACTIONS} successful.")
    print(f"Initial Balance: {initial_balance:.6f} SOL")
    print(f"Final Balance: {final_balance:.6f} SOL")
    print(f"Results saved to: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
