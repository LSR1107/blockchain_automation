import json
import time
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import transfer, TransferParams
from solders.message import Message
from solders.transaction import Transaction
from solana.rpc.api import Client
from typing import cast

# --- Configuration ---
WALLET_FILE = r"C:\Users\lohit\.config\solana\devnet.json"
RPC_URL = "https://api.devnet.solana.com"
RECIPIENT_PUBKEY = Pubkey.from_string("6Mzk8SNy86m8U7NxhcJ4C3GAwFjwNNV4viXEY622JmBt")
NUM_TRANSACTIONS = 20
TRANSFER_AMOUNT_SOL = 0.01  # SOL

# --- Load wallet keypair ---
with open(WALLET_FILE, "r") as f:
    secret = json.load(f)
wallet = Keypair.from_bytes(bytes(secret))

# --- Solana client ---
client = Client(RPC_URL, timeout=30)

print(f"Simulating {NUM_TRANSACTIONS} transactions...\n")

for i in range(1, NUM_TRANSACTIONS + 1):
    try:
        print(f"Transaction {i}:")

        # --- Create transfer instruction ---
        instruction = transfer(
            TransferParams(
                from_pubkey=wallet.pubkey(),
                to_pubkey=RECIPIENT_PUBKEY,
                lamports=int(TRANSFER_AMOUNT_SOL * 1e9),
            )
        )

        # --- Get recent blockhash ---
        recent_blockhash = client.get_latest_blockhash().value.blockhash

        # --- Create message with the payer specified ---
        message = Message.new_with_blockhash(
            [instruction],
            wallet.pubkey(),
            recent_blockhash
        )

        # --- Create transaction from the message ---
        txn = Transaction.new_unsigned(message)

        # --- Sign transaction ---
        txn.sign([wallet], recent_blockhash)

        # --- Send transaction and start timer ---
        start_time = time.time()
        response = client.send_transaction(txn)
        signature = cast(str, response.value)
        print(f"  Signature: {signature}")

        # --- Confirm transaction ---
        confirmed = client.confirm_transaction(signature, commitment="confirmed")
        end_time = time.time()

        status = "Confirmed" if confirmed.value[0].err is None else "Failed"
        confirmation_time = end_time - start_time
        print(f"  Status: {status}")
        print(f"  Time to confirm: {confirmation_time:.3f} seconds")

        # --- Fetch transaction details for actual fee ---
        time.sleep(1)  # Give a moment to index
        tx_info = client.get_transaction(signature, encoding="json", commitment="confirmed")
        actual_fee = None
        if tx_info.value:
            meta = None
            # Look for fee in transaction.meta or value.meta
            if hasattr(tx_info.value.transaction, "meta") and tx_info.value.transaction.meta:
                meta = tx_info.value.transaction.meta
            elif hasattr(tx_info.value, "meta") and tx_info.value.meta:
                meta = tx_info.value.meta

            if meta:
                actual_fee = meta.fee / 1e9

        if actual_fee is not None:
            print(f"  Fee: {actual_fee:.9f} SOL")
        else:
            print("  Could not fetch transaction fee.")

        print()
        time.sleep(1)

    except Exception as e:
        print(f"  Transaction {i} failed: {e}\n")

# --- Final balance ---
final_balance = client.get_balance(wallet.pubkey()).value / 1e9
print(f"Final Balance: {final_balance} SOL")
