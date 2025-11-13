import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";

import Home from "./pages/Home";
import AnalysisDashboard from "./pages/AnalysisDashboard";
import SimulationDashboard from "./pages/SimulationDashboard";
//import ApiSettings from "./pages/ApiSettings";
import AboutProject from "./pages/AboutProject";

import { ThemeProvider } from "@mui/material/styles";
import theme from "./theme";

import { Routes, Route } from "react-router-dom";
import { useState } from "react";

export default function App() {
  const [selectedChain, setSelectedChain] = useState("Ethereum");

  const mockStats = {
    Ethereum: { tps: 52, fee: "23 gwei", block: "18,542,993" },
    Bitcoin: { tps: 7, fee: "12 sat/vB", block: "805,245" },
    Solana: { tps: 2500, fee: "0.00025 SOL", block: "198,550,123" },
    Custom: { tps: "-", fee: "-", block: "-" },
  };

  return (
    <ThemeProvider theme={theme}>
      <div style={{ display: "flex" }}>
        {/* Sidebar fixed left */}
        <Sidebar />

        {/* Main content */}
        <div style={{ flexGrow: 1 }}>
          {/* Topbar fixed at top (full width minus sidebar) */}
          <Topbar
            selectedChain={selectedChain}
            onChangeChain={setSelectedChain}
          />

          {/* Page content */}
          <main
            style={{
              marginTop: "80px", // push below AppBar
              padding: "30px",
              width: "100%",
              boxSizing: "border-box",
              minHeight: "100vh",
              overflowX: "hidden", // prevent sideways scroll
            }}
          >
            <Routes>
              <Route
                path="/"
                element={
                  <Home
                    selectedChain={selectedChain}
                    stats={mockStats[selectedChain]}
                  />
                }
              />
              <Route
                path="/analysis"
                element={<AnalysisDashboard blockchain={selectedChain} />}
              />
              <Route path="/simulation" element={<SimulationDashboard selectedChain={selectedChain} />} />
              {/* <Route path="/api-settings" element={<ApiSettings />} /> */}
              <Route path="/about" element={<AboutProject />} />
            </Routes>
          </main>
        </div>
      </div>
    </ThemeProvider>
  );
}
