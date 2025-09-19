import { Card, CardContent, Typography } from "@mui/material";

export default function RecommendationsPanel({
  action,
  feeRange,
  time,
  confidence,
}) {
  return (
    <Card
      sx={{
        marginTop: 2,
        backgroundColor: "#2a2a2a",
        borderRadius: "12px",
        boxShadow: 3,
      }}
    >
      <CardContent>
        <Typography variant="h6" sx={{ color: "#fff", marginBottom: 1 }}>
          AI Recommendations
        </Typography>
        <Typography sx={{ color: "#ddd" }}>
          Suggested Action: <strong>{action || "N/A"}</strong>
        </Typography>
        <Typography sx={{ color: "#ddd" }}>
          Fee Estimate: {feeRange || "N/A"}
        </Typography>
        <Typography sx={{ color: "#ddd" }}>
          Predicted Confirmation Time: {time ?? "N/A"} seconds
        </Typography>
        <Typography sx={{ color: "#ddd" }}>
          Confidence: {confidence ?? "N/A"}%
        </Typography>
      </CardContent>
    </Card>
  );
}
