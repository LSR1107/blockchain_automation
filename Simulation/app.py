# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from simulate_congestion import simulate_transaction  # Solana simulation
from send_tx import send_btc_tx  # Bitcoin simulation

app = Flask(__name__)
CORS(app)

# --- Example Bitcoin test credentials (replace with your WIF and receiver)
WIF = "cNgiiokb54DcS8XXX92dEQsokqnNQQrsnpJuysG3kMYTN5Dq9V7S"
BTC_RECEIVER = "n3nrdibSbH7sWhbVSZh11MeUk8fFqtLX4W"


@app.route("/api/run_simulation", methods=["POST"])
def run_simulation():
    """
    Example POST body:
    {
      "coin": "solana" | "bitcoin",
      "congestion": "low" | "medium" | "high"
    }
    """
    data = request.json or {}
    coin = data.get("coin", "solana").lower()
    congestion = data.get("congestion", "medium")

    try:
        if coin == "solana":
            result = simulate_transaction(congestion)
            fee = result.get("fee_SOL", 0.0)
            return jsonify({"coin": "solana", "fee_paid": fee})

        elif coin == "bitcoin":
            tx_result = send_btc_tx(WIF, BTC_RECEIVER, amount_btc=0.00001)
            if "fee_sats" in tx_result:
                fee_btc = tx_result["fee_sats"] / 100_000_000
                return jsonify({"coin": "bitcoin", "fee_paid": fee_btc})
            else:
                return jsonify({"coin": "bitcoin", "error": tx_result.get("error", "Transaction failed")})

        else:
            return jsonify({"error": "Unsupported coin type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
