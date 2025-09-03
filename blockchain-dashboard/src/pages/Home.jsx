import { Card, CardContent, Typography, Grid, Box, Button, Container } from "@mui/material";
import { LineChart, Line, ResponsiveContainer, Treemap, Tooltip } from "recharts";
import { Speed, LocalGasStation, Layers } from "@mui/icons-material";
import CountUp from "react-countup";
import { useNavigate } from "react-router-dom";

// Mock TPS trend data
const previewData = [
  { time: "10:00", tps: 40 },
  { time: "10:05", tps: 48 },
  { time: "10:10", tps: 55 },
  { time: "10:15", tps: 50 },
];

// Mock transaction data for treemap
const txData = [
  { name: "Tx1", size: 500, fee: 2.5 },
  { name: "Tx2", size: 200, fee: 0.8 },
  { name: "Tx3", size: 800, fee: 5.2 },
  { name: "Tx4", size: 300, fee: 1.5 },
  { name: "Tx5", size: 150, fee: 0.6 },
  { name: "Tx6", size: 1000, fee: 7.1 },
];

// Custom Tooltip for Treemap
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div
        style={{
          background: "#1E1E1E",
          padding: "8px",
          borderRadius: "6px",
          color: "#fff",
        }}
      >
        <p><strong>{d.name}</strong></p>
        <p>Size: {d.size}</p>
        <p>Fee: {d.fee} sats</p>
      </div>
    );
  }
  return null;
};

export default function Home({ stats = { tps: 52, fee: "23 gwei", block: 18542993 }, selectedChain = "Ethereum" }) {
  const navigate = useNavigate();

  return (
    <Container maxWidth="xl" sx={{ padding: "30px", marginLeft: "150px", color: "white" }}>
      {/* Title */}
      <Typography variant="h4" sx={{ fontWeight: "bold", mb: 2 }}>
        Blockchain Scalability Analyzer – {selectedChain}
      </Typography>

      <Typography variant="body1" sx={{ color: "text.secondary", mb: 4 }}>
        Monitor blockchain metrics, predict congestion, and get AI-powered transaction recommendations.
      </Typography>

      {/* Blockchain Stats */}
      <Grid container spacing={3}>
        {/* TPS */}
        <Grid item xs={12} sm={4}>
          <Card
            sx={{
              background: "linear-gradient(135deg, #1E1E1E, #2C2C3A)",
              borderRadius: "16px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
              "&:hover": { transform: "scale(1.05)", boxShadow: "0 0 20px rgba(0,150,255,0.5)" },
              transition: "all 0.3s ease",
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <Speed fontSize="large" color="success" />
                <Box>
                  <Typography variant="h6">TPS</Typography>
                  <Typography variant="h4" fontWeight="bold">
                    <CountUp end={stats.tps} duration={2} />
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Fee */}
        <Grid item xs={12} sm={4}>
          <Card
            sx={{
              background: "linear-gradient(135deg, #1E1E1E, #332C2C)",
              borderRadius: "16px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
              "&:hover": { transform: "scale(1.05)", boxShadow: "0 0 20px rgba(255,200,0,0.5)" },
              transition: "all 0.3s ease",
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <LocalGasStation fontSize="large" sx={{ color: "#FFD700" }} />
                <Box>
                  <Typography variant="h6">Avg Gas Fee</Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {stats.fee}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Block */}
        <Grid item xs={12} sm={4}>
          <Card
            sx={{
              background: "linear-gradient(135deg, #1E1E1E, #1B2C3A)",
              borderRadius: "16px",
              boxShadow: "0 4px 20px rgba(0,0,0,0.4)",
              "&:hover": { transform: "scale(1.05)", boxShadow: "0 0 20px rgba(0,255,255,0.5)" },
              transition: "all 0.3s ease",
            }}
          >
            <CardContent>
              <Box display="flex" alignItems="center" gap={2}>
                <Layers fontSize="large" sx={{ color: "#00FFFF" }} />
                <Box>
                  <Typography variant="h6">Latest Block</Typography>
                  <Typography variant="h4" fontWeight="bold">
                    {stats.block}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Dashboard Shortcuts */}
      <Grid container spacing={3} sx={{ mt: 4 }}>
        <Grid item xs={12} sm={6}>
          <Card sx={{ backgroundColor: "#2a2a2a", borderRadius: "16px", p: 2, boxShadow: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Analysis Dashboard
              </Typography>
              <Typography sx={{ mt: 1, color: "text.secondary" }}>
                Explore TPS, gas fees, block sizes, and congestion.
              </Typography>
              <Button variant="contained" sx={{ mt: 2 }} onClick={() => navigate("/analysis")}>
                Go
              </Button>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6}>
          <Card sx={{ backgroundColor: "#2a2a2a", borderRadius: "16px", p: 2, boxShadow: 3 }}>
            <CardContent>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                Simulation Dashboard
              </Typography>
              <Typography sx={{ mt: 1, color: "text.secondary" }}>
                Run mock transactions with different priorities.
              </Typography>
              <Button variant="contained" sx={{ mt: 2 }} onClick={() => navigate("/simulation")}>
                Go
              </Button>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Heatmap Treemap */}
      <Card sx={{ mt: 4, background: "#1E1E1E", borderRadius: "16px", p: 2, boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Transaction Heatmap
          </Typography>
          <div style={{ width: "100%", height: 400 }}>
            <ResponsiveContainer>
              <Treemap data={txData} dataKey="size" stroke="#333" fill="#1976d2">
                <Tooltip content={<CustomTooltip />} />
              </Treemap>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      {/* TPS Trend Chart */}
      <Card
        sx={{
          mt: 4,
          backgroundColor: "#1e1e1e",
          borderRadius: "16px",
          p: 3,
          boxShadow: 3,
        }}
      >
        <Typography variant="h6" sx={{ mb: 2 }}>
          TPS Trend (Last 15 min)
        </Typography>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={previewData}>
            <Line type="monotone" dataKey="tps" stroke="#8884d8" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </Container>
  );
}
