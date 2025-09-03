import { useState, useEffect } from "react";
import MetricsPanel from "../components/MetricsPanel";
import RecommendationsPanel from "../components/RecommendationsPanel";
import axios from "axios";

export default function AnalysisDashboard({ blockchain }) {
  const [tpsData, setTpsData] = useState([]);
  const [gasData, setGasData] = useState([]);
  const [blockData, setBlockData] = useState([]);
  const [recommendations, setRecommendations] = useState({});
  const [congestion, setCongestion] = useState(0.4);

  useEffect(() => {
    // Mock TPS data
    setTpsData([
      { time: "10:00", tps: 45 },
      { time: "10:05", tps: 50 },
      { time: "10:10", tps: 55 },
      { time: "10:15", tps: 53 },
    ]);

    // Mock Gas Fee data
    setGasData([
      { time: "10:00", gas: 20 },
      { time: "10:05", gas: 25 },
      { time: "10:10", gas: 22 },
      { time: "10:15", gas: 18 },
    ]);

    // Mock Block Size data
    setBlockData([
      { time: "10:00", size: 1.2 },
      { time: "10:05", size: 1.3 },
      { time: "10:10", size: 1.5 },
      { time: "10:15", size: 1.4 },
    ]);

    // Mock Congestion
    setCongestion(0.6);

    // Mock AI Recommendations
    setRecommendations({
      action: "Execute Now",
      feeRange: "0.00021 - 0.00025 ETH",
      time: 30,
      confidence: 92,
    });
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
