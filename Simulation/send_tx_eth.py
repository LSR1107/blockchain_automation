from web3 import Web3
from eth_account import Account

# --- Setup ---
# Use your own Sepolia RPC endpoint here
RPC_URL = "https://sepolia.infura.io/v3/573cbc188608432f842e2c494a6bae5c"
w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    raise Exception("Cannot connect to Ethereum network")

# --- Step 1: Create new sender & receiver addresses ---
sender_acct = Account.create()
receiver_acct = Account.create()

print("Sender Address:", sender_acct.address)
print("Sender Private Key:", sender_acct.key.hex())
print("Receiver Address:", receiver_acct.address)

print("\n⚠️ Send some Sepolia ETH to the sender address before continuing.\n")

# Wait until you have funded the sender, then continue manually below

# --- Step 2: Build and send a transaction ---
# Replace with the funded sender address/private key if needed
sender_address = sender_acct.address
private_key = sender_acct.key

# Check balance
balance = w3.eth.get_balance(sender_address)
print("Sender balance (wei):", balance)

if balance == 0:
    raise Exception("Sender has 0 balance. Fund it with Sepolia ETH first.")

# Transaction parameters
nonce = w3.eth.get_transaction_count(sender_address)
gas_price = w3.eth.gas_price

tx = {
    'nonce': nonce,
    'to': receiver_acct.address,
    'value': w3.to_wei(0.0001, 'ether'),  # send 0.0001 ETH
    'gas': 21000,
    'gasPrice': gas_price,
    'chainId': 11155111  # Sepolia chain ID
}

# Sign the transaction
signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)

# Broadcast
tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
print("Transaction sent! TX hash:", w3.to_hex(tx_hash))

# Wait for receipt (optional)
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print("Tx receipt:", receipt)
