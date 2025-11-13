import { useState, useEffect } from "react";
import MetricsPanel from "../components/MetricsPanel";
import RecommendationsPanel from "../components/RecommendationsPanel";
import GaugeChart from "react-gauge-chart";
import axios from "axios";

const BACKEND_URL = " https://c2a3afd9bd4b.ngrok-free.app"; // 👈 FastAPI backend

export default function AnalysisDashboard({ blockchain }) {
  const [tpsData, setTpsData] = useState([]);
  const [gasData, setGasData] = useState([]);
  const [blockData, setBlockData] = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const [congestion, setCongestion] = useState(0);

  useEffect(() => {
    async function fetchData() {
      try {
        // Common headers for ngrok
        const headers = { "ngrok-skip-browser-warning": "true" };

        // ✅ fetch recent blocks
        const res = await axios.get(
          `${BACKEND_URL}/metrics/${blockchain}/live/recent?n=20`,
          { headers }
        );
        const blocks = res.data;

        if (!Array.isArray(blocks)) {
          console.error("Backend did not return an array:", blocks);
          return;
        }

        // ✅ TPS chart
        setTpsData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            tps: b.tps_estimate,
          }))
        );

        // ✅ Gas / Fees (Ethereum vs Solana)
        setGasData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            gas:
              blockchain === "Ethereum"
                ? b.avg_fee_per_tx_eth
                : b.avg_fee_per_tx / 1e9, // lamports → SOL
          }))
        );

        // ✅ Block size chart
        setBlockData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            size: b.tx_count, // or block size if backend provides
          }))
        );

        // ✅ latest block for recommendations + congestion - FIX: Added missing headers
        const latest = await axios.get(
          `${BACKEND_URL}/metrics/${blockchain}/live/latest`,
          { headers } // 👈 This was missing!
        );
        const lb = latest.data;

        console.log("Latest Block Data:", lb);

        // ✅ Congestion calculation - Added safety checks
        let tps = lb?.tps_estimate ?? 0;
        let maxTPS = blockchain === "Ethereum" ? 30 : 5000;
        let congestionValue = tps && maxTPS ? Math.min(tps / maxTPS, 1) : 0;
        setCongestion(congestionValue);

        // ✅ Fee calculation - Added better error handling
        let feeRange = "N/A";
        if (blockchain === "Ethereum") {
          // already ETH
          let feeEth = lb?.avg_fee_per_tx_eth ?? 0;
          feeRange = feeEth > 0 ? `${feeEth.toFixed(6)} ETH` : "N/A";
        } else if (blockchain === "Solana") {
          // lamports → SOL
          let feeLamports = lb?.avg_fee_per_tx ?? 0;
          feeRange = feeLamports > 0 ? `${(feeLamports / 1e9).toFixed(9)} SOL` : "N/A";
        }

        // ✅ AI Recommendations - Improved logic
        setRecommendations({
          action: congestionValue > 0.7 ? "Wait" : "Execute Now",
          feeRange,
          time: blockchain === "Ethereum" ? 30 : 5, // prediction window
          confidence: congestionValue > 0 ? (100 - congestionValue * 100).toFixed(1) : "N/A", // better confidence calc
        });
      } catch (err) {
        console.error("Error fetching data:", err);
        
        // Set fallback values on error
        setRecommendations({
          action: "Data Unavailable",
          feeRange: "N/A",
          time: "N/A",
          confidence: "N/A"
        });
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, [blockchain]);

  return (
    <div style={{ padding: "20px" }}>
      {/* Metrics panel: TPS, Gas, Block size */}
      <MetricsPanel
        tpsData={tpsData}
        gasData={gasData}
        blockData={blockData}
        congestion={congestion}
      />



      {/* AI Recommendations */}
      <RecommendationsPanel {...recommendations} />
    </div>
  );
}