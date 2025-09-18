import { useState, useEffect } from "react";
import MetricsPanel from "../components/MetricsPanel";
import RecommendationsPanel from "../components/RecommendationsPanel";
import axios from "axios";

const BACKEND_URL = "http://127.0.0.1:8000"; // 👈 FastAPI

export default function AnalysisDashboard({ blockchain }) {
  const [tpsData, setTpsData] = useState([]);
  const [gasData, setGasData] = useState([]);
  const [blockData, setBlockData] = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const [congestion, setCongestion] = useState(0.4);

  useEffect(() => {
    async function fetchData() {
      try {
        // fetch recent blocks
        const res = await axios.get(`${BACKEND_URL}/metrics/live/recent?n=20`);
        const blocks = res.data;

        // Transform into recharts format
        setTpsData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            tps: b.tps_estimate,
          }))
        );

        setGasData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            gas: b.avg_fee_per_tx,
          }))
        );

        setBlockData(
          blocks.map((b) => ({
            time: new Date(b.timestamp).toLocaleTimeString(),
            size: b.tx_count, // or replace with block size if API gives
          }))
        );

        // latest block (for congestion + recommendations)
        const latest = await axios.get(`${BACKEND_URL}/metrics/live/latest`);
        const lb = latest.data;

        setCongestion(Math.min(lb.tps_estimate / 5000, 1)); // crude estimate
        setRecommendations({
          action: lb.tps_estimate > 2000 ? "Wait" : "Execute Now",
          feeRange: `${lb.avg_fee_per_tx} lamports`,
          time: 30,
          confidence: 85,
        });
      } catch (err) {
        console.error("Error fetching data:", err);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 10000); // refresh every 10s
    return () => clearInterval(interval);
  }, [blockchain]);

  return (
    <div style={{ padding: "20px" }}>
      <MetricsPanel
        tpsData={tpsData}
        gasData={gasData}
        blockData={blockData}
        congestion={congestion}
      />
      <RecommendationsPanel {...recommendations} />
    </div>
  );
}
