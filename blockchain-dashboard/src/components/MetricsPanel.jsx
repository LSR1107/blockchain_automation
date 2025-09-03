import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  AreaChart,
  Area,
  BarChart,
  Bar,
  ResponsiveContainer,
} from "recharts";
import GaugeChart from "react-gauge-chart";
import { Typography } from "@mui/material";

export default function MetricsPanel({ tpsData, gasData, blockData, congestion }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(2, 1fr)",
        gap: "20px",
        padding: "20px",
        width: "100%",
        boxSizing: "border-box",
      }}
    >
      {/* TPS Line Chart */}
      <div
        style={{
          background: "#1e1e1e",
          padding: "15px",
          borderRadius: "12px",
          width: "100%",
          minHeight: "360px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h6" color="white">
          Transactions Per Second
        </Typography>
        <Typography variant="body2" color="#aaa" style={{ marginBottom: "10px" }}>
          Shows how many transactions are processed by the network over time.
        </Typography>
        <div style={{ flexGrow: 1 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={tpsData}>
              <Line type="monotone" dataKey="tps" stroke="#8884d8" strokeWidth={2} />
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="time" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Gas Fee Area Chart */}
      <div
        style={{
          background: "#1e1e1e",
          padding: "15px",
          borderRadius: "12px",
          width: "100%",
          minHeight: "360px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h6" color="white">
          Gas Fees
        </Typography>
        <Typography variant="body2" color="#aaa" style={{ marginBottom: "10px" }}>
          Average gas fee paid per transaction across time intervals.
        </Typography>
        <div style={{ flexGrow: 1 }}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={gasData}>
              <Area type="monotone" dataKey="gas" stroke="#82ca9d" fill="#82ca9d" />
              <XAxis dataKey="time" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Block Size Bar Chart */}
      <div
        style={{
          background: "#1e1e1e",
          padding: "15px",
          borderRadius: "12px",
          width: "100%",
          minHeight: "360px",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h6" color="white">
          Block Size
        </Typography>
        <Typography variant="body2" color="#aaa" style={{ marginBottom: "10px" }}>
          Visualizes the size of blocks mined on the network over time.
        </Typography>
        <div style={{ flexGrow: 1 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={blockData}>
              <Bar dataKey="size" fill="#ffc658" />
              <XAxis dataKey="time" stroke="#aaa" />
              <YAxis stroke="#aaa" />
              <Tooltip />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Network Congestion Gauge */}
      <div
        style={{
          background: "#1e1e1e",
          padding: "20px",
          borderRadius: "12px",
          width: "100%",
          minHeight: "360px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Typography variant="h6" color="white">
          Network Congestion
        </Typography>
        <Typography variant="body2" color="#aaa" style={{ marginBottom: "10px" }}>
          Indicates how busy the network is at the current moment.
        </Typography>
        <GaugeChart id="gauge-chart" nrOfLevels={20} percent={congestion} textColor="#fff" />
      </div>
    </div>
  );
}
