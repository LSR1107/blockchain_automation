import subprocess
import sys
import os
import json

def run_simulation():
    # Step 1: Run tx_creator.py and get JSON output
    tx_creator_path = os.path.join("Simulation", "send_tx.py")
    tx_output = subprocess.check_output(
        [sys.executable, tx_creator_path],
        text=True
    ).strip()

    # Expect JSON from tx_creator
    tx_info = json.loads(tx_output)
    #txid = tx_info.get("txid")

    # Step 2: Run PowerShell script with txid
    ps1_path = os.path.join("Simulation", "check_tx_fee.ps1")
    result = subprocess.check_output(
        ["pwsh", "-File", ps1_path, "-Txid", tx_info["txid"]],
        text=True
    ).strip()

    return {
        "transaction": tx_info,
        "checker_result": result
    }


if __name__ == "__main__":
    print(run_simulation())
