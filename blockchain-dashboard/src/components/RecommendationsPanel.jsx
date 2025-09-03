import { Card, CardContent, Typography } from "@mui/material";

export default function RecommendationsPanel({ action, feeRange, time, confidence }) {
  return (
    <Card  sx={{ 
                marginTop: 2, 
                backgroundColor: "#2a2a2a", 
                borderRadius: "12px", 
                boxShadow: 3 
  }}>
      <CardContent>
        <Typography variant="h6">AI Recommendations</Typography>
        <Typography>
          Suggested Action: <strong>{action}</strong>
        </Typography>
        <Typography>Fee Estimate: {feeRange}</Typography>
        <Typography>Predicted Confirmation Time: {time} seconds</Typography>
        <Typography>Confidence: {confidence}%</Typography>
      </CardContent>
    </Card>
  );
}
