import { Typography } from "@mui/material";

export default function AboutProject() {
  return (
    <div style={{ padding: "20px" }}>
      <Typography variant="h4" gutterBottom>
        About This Project
      </Typography>
      <Typography variant="body1">
        This project analyzes blockchain scalability issues such as congestion, 
        high transaction fees, and network delays. It leverages AI models including 
        GNN, LSTM, and Deep RL to provide actionable recommendations.
      </Typography>
    </div>
  );
}
