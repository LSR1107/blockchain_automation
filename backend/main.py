from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
import torch
from fastapi.responses import JSONResponse
import numpy as np
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
from backend.solana_api import get_latest_block_info, get_recent_blocks
from fastapi.middleware.cors import CORSMiddleware
from backend.simulation_runner import run_simulation
from backend.etherium_api import get_latest_eth_block_info, get_recent_eth_blocks
from backend.bitcoin_api import (
    get_latest_btc_block_info,
    get_recent_btc_blocks
)
from Analysis_backend.ETH_GNN_results import run_eth_gnn_analysis
from Analysis_backend.ETH_ALSTM_results import run_eth_alstm_analysis
from Analysis_backend.BTC_ALSTM_results import run_btc_alstm_analysis
from Analysis_backend.BTC_GNN_results import run_btc_gnn_analysis
from backend.crypto_price import get_prices


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
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


@app.get("/prices")
def live_prices(vs_currency: str = "usd"):
    """
    Returns latest prices of BTC, ETH, SOL in given fiat (usd by default).
    """
    symbols = ["bitcoin", "ethereum", "solana"]
    prices = get_prices(symbols, vs_currency=vs_currency.lower())
    return {"prices": prices}


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
def btc_gnn_analysis(future_steps: int = 10):
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
    


    # api.py





from Analysis_backend.multi_chain_A2C import (
    ChainEnv,
    ActorCriticNet,
    decide_now,
    align_and_fuse_modality_arrays,
    align_labels_to_length
)


# ----------------------------------------
# LOAD DATA
# ----------------------------------------

eth_alstm = np.load("val_embeddings.npy")
eth_gnn   = np.load("ETH_GNN_val_embeddings.npy")
eth_y     = np.load("y_val.npy")           # WEI PER GAS ✅

btc_alstm = np.load("btc_val_embeddings.npy")
btc_gnn   = np.load("Analysis_backend/saved_outputs/val_embeddings.npy")
btc_y     = np.load("btc_y_val.npy")       # BTC ✅

eth_fused, L1 = align_and_fuse_modality_arrays(eth_alstm, eth_gnn)
btc_fused, L2 = align_and_fuse_modality_arrays(btc_alstm, btc_gnn)

eth_y = align_labels_to_length(eth_y, L1)
btc_y = align_labels_to_length(btc_y, L2)


# ----------------------------------------
# CREATE ENVIRONMENTS
# ----------------------------------------

eth_env = ChainEnv(eth_fused, eth_y, chain="ETH")
btc_env = ChainEnv(btc_fused, btc_y, chain="BTC")


# ----------------------------------------
# LOAD TRAINED MODELS
# ----------------------------------------

eth_agent  = ActorCriticNet(eth_env.D + 3)
btc_agent  = ActorCriticNet(btc_env.D + 3)
meta_agent = ActorCriticNet((eth_env.D + 3) + (btc_env.D + 3), n_actions=2)

eth_agent.load_state_dict(torch.load("eth_agent.pth", map_location="cpu"))
btc_agent.load_state_dict(torch.load("btc_agent.pth", map_location="cpu"))
meta_agent.load_state_dict(torch.load("meta_agent.pth", map_location="cpu"))

eth_agent.eval()
btc_agent.eval()
meta_agent.eval()

GAS_MAP = {
    "transfer": 21000,
    "erc20": 65000,
    "nft": 120000,
    "contract": 150000
}

# ==============================
# INPUT MODELS
# ==============================

class PriceInput(BaseModel):
    eth_usd: float
    btc_usd: float


class FinalizeInput(BaseModel):
    eth_usd: float = 3100
    btc_usd: float = 99000
    user_preference: str  # eth / btc / neutral

    # ETH fee calculation
    eth_tx_type: str | None = None   # low/medium/high/custom
    eth_custom_gas: int | None = None
    eth_custom_gwei: float | None = None

    # BTC fee calculation
    btc_mode: str | None = None  # native / vbytes
    btc_custom_value: float | None = None  # BTC
    btc_inputs: int | None = None
    btc_outputs: int | None = None
    btc_sat_per_vb: float | None = None


# =====================================
# UTILITIES
# =====================================

GAS_MAP = {
    "low": 21000,
    "medium": 100000,
    "high": 300000
}


def calculate_eth_total_fee(predicted_gwei, gas_units, eth_price):
    eth_value = (predicted_gwei * gas_units) / 1e9
    usd_value = eth_value * eth_price
    return float(eth_value), float(usd_value)

def calculate_btc_total_fee(predicted_btc, btc_price):
    return float(predicted_btc), float(predicted_btc * btc_price)

def eth_fee_from_user(gwei_price, gas_units, eth_usd):
    eth_fee = gwei_price * 1e-9 * gas_units
    usd = eth_fee * eth_usd
    return eth_fee, usd

def btc_fee_from_native(btc_fee, btc_usd):
    usd = btc_fee * btc_usd
    return btc_fee, usd

def btc_fee_from_vbytes(n_inputs, n_outputs, sat_per_vb, btc_usd):
    size = 10 + n_inputs * 148 + n_outputs * 34
    sats = size * sat_per_vb
    btc_fee = sats * 1e-8
    usd = btc_fee * btc_usd
    return btc_fee, usd, size, sats



# =====================================
# /PREDICT
# =====================================

@app.post("/predict")
def predict(data: PriceInput):
    try:
        result = decide_now(
            eth_env,
            btc_env,
            eth_agent,
            btc_agent,
            meta_agent,
            eth_usd_price=data.eth_usd,
            btc_usd_price=data.btc_usd
        )

        return {
            "recommended_chain": result["recommended_chain"],
            "recommended_action": result["recommended_action"],

            "chain_probabilities": result["chain_probabilities"].tolist(),
            "action_probabilities": result["action_probabilities"].tolist(),

            "predicted_eth_gwei": result["predicted_eth_gwei"],
            "predicted_btc_fee": result["predicted_btc_fee"],

            # ✅ REAL trend signal
            "native_delta": result["native_delta"],
            "volatility": result["volatility"],
            "current_native_value": result["current_native_value"],

            # ✅ CORRECT KEY
            "expected_confirmation_delay": result["confirmation_delay_sec"],
            "expected_reward": result["expected_reward"],

            "explanation": result["explanation"]
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/finalize")
def finalize(data: FinalizeInput):

    # -------------------------------------------------
    # 1. RUN META + SUB AGENTS (same as /predict)
    # -------------------------------------------------
    result = decide_now(
        eth_env,
        btc_env,
        eth_agent,
        btc_agent,
        meta_agent,
        eth_usd_price=data.eth_usd,
        btc_usd_price=data.btc_usd
    )

    original_chain = result["recommended_chain"]
    final_chain = original_chain

    # -------------------------------------------------
    # 2. APPLY USER PREFERENCE
    # -------------------------------------------------
    pref = data.user_preference.lower()
    if pref == "eth":
        final_chain = "ETH"
    elif pref == "btc":
        final_chain = "BTC"

    # -------------------------------------------------
    # 3. PREPARE SHARED MODEL REASON STRING
    # -------------------------------------------------
    model_reason = (
        f"Model reason: meta-agent probabilities = {result['chain_probabilities']}. "
        f"Sub-agent action probabilities = {result['action_probabilities']}. "
        f"The model picked {original_chain} because the expected short-term reward "
        f"(critic={result['expected_reward']:.3f}) was highest."
    )

    # -------------------------------------------------
    # 4. IF FINAL CHOICE = ETH
    # -------------------------------------------------
    if final_chain == "ETH":

        predicted_gwei = result["predicted_eth_gwei"]
        if predicted_gwei is None:
            return {"error": "ETH prediction unavailable"}

        # --- Select gas units based on tx type ---
        if data.eth_tx_type in GAS_MAP:
            gas_units = GAS_MAP[data.eth_tx_type]

        elif data.eth_tx_type == "custom" and data.eth_custom_gas is not None:
            gas_units = data.eth_custom_gas

        else:
            gas_units = 21000  # fallback

        # --- User-finalized gas fee ---
        eth_fee_eth, eth_fee_usd = eth_fee_from_user(
            predicted_gwei,
            gas_units,
            data.eth_usd
        )

        # --- Alternative BTC estimate ---
        alt_btc_usd = None
        if result.get("predicted_btc_fee"):
            _, alt_btc_usd = calculate_btc_total_fee(
                result["predicted_btc_fee"],
                data.btc_usd
            )

        # --- Suggestion based on model trend ---
        suggestion = (
            "✅ Ethereum selected. Good time to send."
            if result["native_delta"] > 0 else
            "⚠️ Gas predicted to drop. Consider waiting 10–20 minutes."
        )

        # --- Terminal-style personalized summary ---
        personalized_summary = (
            "=== PERSONALIZED SUMMARY ===\n"
            f"User preference: prefer_eth\n"
            f"Ethereum estimated total fee: ${eth_fee_usd:.6f} USD\n"
        )

        if alt_btc_usd is not None:
            personalized_summary += f"Bitcoin estimated total fee: ${alt_btc_usd:.6f} USD\n"

            cheaper = "ETH" if eth_fee_usd < alt_btc_usd else "BTC"
            personalized_summary += f"Cheaper according to your inputs: {cheaper}\n\n"

        personalized_summary += (
            "💬 User-Based Suggestion:\n"
            f"{suggestion}\n\n"
            f"Reason: {model_reason}"
        )

        return {
            "original_choice": original_chain,
            "final_choice": "ETH",

            "predicted_gas_gwei": predicted_gwei,
            "gas_units_used": gas_units,

            "final_fee_eth": eth_fee_eth,
            "final_fee_usd": eth_fee_usd,

            "expected_confirmation_delay": result["confirmation_delay_sec"],

            "alternative_chain": "BTC",
            "alternative_usd_cost": alt_btc_usd,

            "suggestion": suggestion,
            "model_explanation": result["explanation"],

            "personalized_summary": personalized_summary
        }

    # -------------------------------------------------
    # 5. IF FINAL CHOICE = BTC
    # -------------------------------------------------
    else:

        predicted_btc = result.get("predicted_btc_fee")
        if predicted_btc is None:
            return {"error": "BTC prediction unavailable"}

        # --- BTC USER MODE ---
        btc_fee_btc = None
        btc_fee_usd = None
        tx_size = None
        sats = None

        # MODE A: BTC NATIVE VALUE PROVIDED
        if data.btc_mode == "native" and data.btc_custom_value:
            btc_fee_btc, btc_fee_usd = btc_fee_from_native(
                data.btc_custom_value,
                data.btc_usd
            )

        # MODE B: VBYTE MODE (inputs, outputs)
        elif data.btc_mode == "vbytes" and data.btc_custom_value is not None:

            # btc_custom_value = sat/vB
            sat_vb = data.btc_custom_value

            # ask for input/output - here simplified default
            n_inputs = 2
            n_outputs = 2

            btc_fee_btc, btc_fee_usd, tx_size, sats = btc_fee_from_vbytes(
                n_inputs,
                n_outputs,
                sat_vb,
                data.btc_usd
            )

        # Default fallback: use model-predicted BTC fee
        else:
            btc_fee_btc, btc_fee_usd = calculate_btc_total_fee(
                predicted_btc,
                data.btc_usd
            )

        # --- Alternative ETH estimate ---
        alt_eth_usd = None
        if result.get("predicted_eth_gwei"):
            _, alt_eth_usd = calculate_eth_total_fee(
                result["predicted_eth_gwei"],
                21000,
                data.eth_usd
            )

        suggestion = (
            "✅ Bitcoin is efficient now."
            if result["native_delta"] > 0 else
            "⚠️ Fees may reduce soon. Try again in 15 minutes."
        )

        # --- Terminal-style summary ---
        personalized_summary = (
            "=== PERSONALIZED SUMMARY ===\n"
            f"User preference: prefer_btc\n"
            f"Bitcoin estimated total fee: ${btc_fee_usd:.6f} USD\n"
        )

        if alt_eth_usd is not None:
            personalized_summary += (
                f"Ethereum estimated total fee: ${alt_eth_usd:.6f} USD\n"
            )

            cheaper = "BTC" if btc_fee_usd < alt_eth_usd else "ETH"
            personalized_summary += f"Cheaper according to your inputs: {cheaper}\n\n"

        personalized_summary += (
            "💬 User-Based Suggestion:\n"
            f"{suggestion}\n\n"
            f"Reason: {model_reason}"
        )

        return {
            "original_choice": original_chain,
            "final_choice": "BTC",

            "final_fee_btc": btc_fee_btc,
            "final_fee_usd": btc_fee_usd,

            "expected_confirmation_delay": result["confirmation_delay_sec"],

            "alternative_chain": "ETH",
            "alternative_usd_cost": alt_eth_usd,

            "suggestion": suggestion,
            "model_explanation": result["explanation"],

            "personalized_summary": personalized_summary
        }
