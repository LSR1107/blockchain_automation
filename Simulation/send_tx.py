# send_tx.py
from bit import PrivateKeyTestnet
import requests, math, sys
import json

def send_btc_tx(WIF, receiver, amount_btc=0.001):
    sender_address = PrivateKeyTestnet(WIF).address

    # --- Fetch UTXOs ---
    utxos_url = f"https://blockstream.info/testnet/api/address/{sender_address}/utxo"
    utxos = requests.get(utxos_url).json()
    if not utxos:
        return {"error": "❌ No UTXOs found - fund this address first"}

    total_sats = sum(u['value'] for u in utxos)

    # --- Fee estimation ---
    fee_data = requests.get("https://blockstream.info/testnet/api/fee-estimates").json()
    feerate = fee_data.get("1", 1.5)  # sat/vB, target = 1 block

    estimated_vsize = 148 + 34*2 + 10   # ≈ 226 vbytes
    fee_sats = max(math.ceil(feerate * estimated_vsize), 226)  # enforce min relay fee

    send_sats = int(amount_btc * 100_000_000)
    if send_sats + fee_sats > total_sats:
        return {"error": "❌ Not enough balance to cover amount + fee"}

    # --- Create and sign transaction ---
    key = PrivateKeyTestnet(WIF)
    tx_hex = key.create_transaction(
        [(receiver, amount_btc, 'btc')],
        fee=fee_sats,
        absolute_fee=True
    )

    # --- Broadcast transaction ---
    resp = requests.post(
        "https://blockstream.info/testnet/api/tx",
        data=tx_hex,
        headers={'Content-Type': 'text/plain'}
    )

    if resp.status_code == 200:
        txid = resp.text.strip()
        return {
            "txid": txid,
            "sender": sender_address,
            "receiver": receiver,
            "amount_btc": amount_btc,
            "fee_sats": fee_sats,
            "feerate_sat_vb": feerate,
            "tx_hex": tx_hex
        }
    else:
        return {
            "error": f"Broadcast failed: {resp.status_code}",
            "details": resp.text
        }

# Run standalone
if __name__ == "__main__":
    WIF = "cNgiiokb54DcS8XXX92dEQsokqnNQQrsnpJuysG3kMYTN5Dq9V7S"
    receiver = "n3nrdibSbH7sWhbVSZh11MeUk8fFqtLX4W"
    result = send_btc_tx(WIF, receiver, amount_btc=0.00001)
    print(json.dumps(result))
