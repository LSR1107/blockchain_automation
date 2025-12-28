import requests
from fastapi import HTTPException

COINGECKO_API = "https://api.coingecko.com/api/v3/simple/price"

def get_prices(symbols: list, vs_currency: str = "usd"):
    """
    symbols: list of coin IDs according to CoinGecko — e.g. ["bitcoin", "ethereum", "solana"]
    vs_currency: fiat — e.g. "usd", "inr"
    """
    ids = ",".join(symbols)
    params = {
        "ids": ids,
        "vs_currencies": vs_currency
    }
    try:
        resp = requests.get(COINGECKO_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return { sym: data.get(sym, {}).get(vs_currency) for sym in symbols }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price API error: {e}")
