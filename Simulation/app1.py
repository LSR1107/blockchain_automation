# app1.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from simulate_congestion import simulate_transaction

app = Flask(__name__)
CORS(app)

@app.route("/api/run_simulation", methods=["POST"])
def run_simulation():
    """
    Example POST body:
    { "congestion": "high" }
    """
    data = request.json or {}
    congestion = data.get("congestion", "medium")
    result = simulate_transaction(congestion)
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
