from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import torch
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from backend.solana_api import get_latest_block_info, get_recent_blocks
from fastapi.middleware.cors import CORSMiddleware
from backend.simulation_runner import run_simulation
from backend.etherium_api import get_latest_eth_block_info, get_recent_eth_blocks
from backend.bitcoin_api import get_latest_btc_block_info, get_recent_btc_blocks
from Analysis_backend.ETH_GNN_results import run_eth_gnn_analysis
from Analysis_backend.ETH_ALSTM_results import run_eth_alstm_analysis
from Analysis_backend.BTC_ALSTM_results import run_btc_alstm_analysis
from Analysis_backend.BTC_GNN_results import run_btc_gnn_analysis


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this to your frontend URL instead of "*"
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)


app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# --- MongoDB connection ---
MONGO_URI = "mongodb://localhost:27017"   
DB_NAME = "solana_db"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]

@app.get("/")
def root():
    return {"message": "🚀 FastAPI backend is running!"}

"""Solana and Etherium live data is being collected here"""
@app.get("/metrics/{chain}/live/latest")
def latest_block(chain: str):
    if chain.lower() == "solana":
        return get_latest_block_info() or {"error": "Solana block fetch failed"}
    elif chain.lower() == "ethereum":
        return get_latest_eth_block_info() or {"error": "Ethereum block fetch failed"}
    elif chain.lower() == "bitcoin":
        return get_latest_btc_block_info() or {"error": "Bitcoin block fetch failed"}
    else:
        return {"error": f"Unsupported chain: {chain}"}



@app.get("/metrics/{chain}/live/recent")
def recent_blocks(chain: str, n: int = 50):
    if chain.lower() == "solana":
        return get_recent_blocks(n_blocks=n)
    elif chain.lower() == "ethereum":
        return get_recent_eth_blocks(n_blocks=n)
    elif chain.lower() == "bitcoin":
        return get_recent_btc_blocks(n_blocks=n)
    else:
        return {"error": f"Unsupported chain: {chain}"}



""" This is the simulation part connecting the transactions """
@app.get("/simulate/btc")
def simulate_btc_tx():
    result = run_simulation()
    return result



""" This is the analysis part of ETH """
@app.get("/analysis/eth/gnn")
def eth_gnn_analysis(future_steps: int = 10):
    """
    Run Ethereum congestion prediction using pretrained GNN model
    Example: /analysis/eth/gnn?future_steps=10
    """
    try:
        result = run_eth_gnn_analysis(future_steps=future_steps)
        #return JSONResponse(content=json.loads(result))
        return result
    except Exception as e:
        return {"error": str(e)}

@app.get("/metrics/eth/alstm-analysis")
def eth_alstm_analysis():
    """
    Run Ethereum Attention-LSTM gas price analysis.
    """
    return run_eth_alstm_analysis()
    

""" This is the analysis part of BTC """
@app.get("/metrics/btc/alstm-analysis")
def btc_alstm_analysis():
    """
    Run Ethereum Attention-LSTM gas price analysis.
    """
    return run_btc_alstm_analysis()

@app.get("/analysis/btc/gnn")
def eth_gnn_analysis(future_steps: int = 10):
    """
    Run Ethereum congestion prediction using pretrained GNN model
    Example: /analysis/eth/gnn?future_steps=10
    """
    try:
        result = run_btc_gnn_analysis(future_steps=future_steps)
        #return JSONResponse(content=json.loads(result))
        return result
    except Exception as e:
        return {"error": str(e)}
    


"""this is the part which is connected to mongodb we will continue tomorrow."""

@app.get("/metrics/summary")
async def metrics_summary():
    """Overall stats"""
    txns = db.transactions
    count = await txns.count_documents({})
    latest_txn = await txns.find_one(sort=[("blockTime", -1)])
    avg_fee = await txns.aggregate([
        {"$group": {"_id": None, "avgFee": {"$avg": "$fee"}}}
    ]).to_list(1)

    return {
        "total_transactions": count,
        "latest_txn": latest_txn,
        "avg_fee": avg_fee[0]["avgFee"] if avg_fee else 0
    }


@app.get("/metrics/tps")
async def metrics_tps(last_minutes: int = 60):
    """Transactions per second over last X minutes"""
    txns = db.transactions
    cutoff = int((datetime.utcnow() - timedelta(minutes=last_minutes)).timestamp())

    pipeline = [
        {"$match": {"blockTime": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": {
                    "$toDate": {"$multiply": ["$blockTime", 1000]} 
                },
                "tx_count": {"$sum": 1}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    result = await txns.aggregate(pipeline).to_list(None)
    return result


@app.get("/metrics/fees")
async def metrics_fees(last_minutes: int = 60):
    """Average fees over last X minutes"""
    txns = db.transactions
    cutoff = int((datetime.utcnow() - timedelta(minutes=last_minutes)).timestamp())

    pipeline = [
        {"$match": {"blockTime": {"$gte": cutoff}}},
        {
            "$group": {
                "_id": {
                    "$toDate": {"$multiply": ["$blockTime", 1000]}
                },
                "avg_fee": {"$avg": "$fee"},
                "max_fee": {"$max": "$fee"},
                "min_fee": {"$min": "$fee"}
            }
        },
        {"$sort": {"_id": 1}}
    ]
    result = await txns.aggregate(pipeline).to_list(None)
    return result



@app.get("/metrics/congestion")
async def congestion(last_minutes: int = 30):
    """
    Approx congestion: ratio of failed txns to total over time
    """
    since = int(datetime.utcnow().timestamp()) - last_minutes * 60
    pipeline = [
        {"$match": {"blockTime": {"$gte": since}}},
        {"$group": {
            "_id": "$blockTime",
            "total": {"$sum": 1},
            "failed": {"$sum": {"$cond": [{"$ne": ["$err", None]}, 1, 0]}}
        }},
        {"$sort": {"_id": 1}}
    ]
    data = await db.transactions.aggregate(pipeline).to_list(None)

    results = []
    for d in data:
        fail_rate = d["failed"] / d["total"] if d["total"] > 0 else 0
        results.append({
            "time": datetime.utcfromtimestamp(d["_id"]).isoformat(),
            "congestion": fail_rate
        })
    return JSONResponse(content=results)

@app.get("/predict")
async def predict_next():
    """
    Run the trained A-LSTM model and return next predictions.
    """
    # TODO: load your trained model checkpoint here
    # Example dummy output for now
    prediction = {
        "next_minute": {"fee": 0.00024, "tx_count": 155},
        "next_hour": {"fee": 0.00028, "tx_count": 8420}
    }
    return prediction
