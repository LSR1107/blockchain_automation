from bit import PrivateKeyTestnet
import requests, math, sys

# --- Sender & Receiver ---
WIF = "cNgiiokb54DcS8XXX92dEQsokqnNQQrsnpJuysG3kMYTN5Dq9V7S"
sender_address = PrivateKeyTestnet(WIF).address
receiver = "n3nrdibSbH7sWhbVSZh11MeUk8fFqtLX4W"
amount_btc = 0.001  # BTC to send

# --- Check UTXOs / balance ---
utxos_url = f"https://blockstream.info/testnet/api/address/{sender_address}/utxo"
utxos = requests.get(utxos_url).json()
if not utxos:
    print("❌ No UTXOs found - fund this address first")
    sys.exit()

total_sats = sum(u['value'] for u in utxos)
print(f"UTXO total: {total_sats} sats")

fee_data = requests.get("https://blockstream.info/testnet/api/fee-estimates").json()
feerate = fee_data.get("1", 1.5)  # sat/vB, target=1 block

# Estimate size: 1 input (~148 vB) + 2 outputs (~34 vB each) + overhead (~10)
estimated_vsize = 148 + 34*2 + 10  # ≈ 226 vbytes

fee_sats = max(math.ceil(feerate * estimated_vsize), 226)  # enforce min relay fee
print(f"Using fee rate ≈ {feerate} sat/vB  →  fee ≈ {fee_sats} sats")

# --- Check if enough balance ---
send_sats = int(amount_btc * 100_000_000)
if send_sats + fee_sats > total_sats:
    print("❌ Not enough balance to cover amount + fee")
    sys.exit()

# --- Create and sign transaction ---
key = PrivateKeyTestnet(WIF)
tx_hex = key.create_transaction(
    [(receiver, amount_btc, 'btc')],
    fee=fee_sats,
    absolute_fee=True
)

# --- Broadcast to Blockstream ---
resp = requests.post(
    "https://blockstream.info/testnet/api/tx",
    data=tx_hex,
    headers={'Content-Type': 'text/plain'}
)

print("TX hex:", tx_hex)
print("Broadcast status:", resp.status_code, resp.text)
