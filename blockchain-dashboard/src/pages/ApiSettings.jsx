import { TextField, Button, Typography } from "@mui/material";
import { useState } from "react";

export default function ApiSettings() {
  const [etherscanKey, setEtherscanKey] = useState("");

  return (
    <div style={{ padding: "20px" }}>
      <Typography variant="h5" gutterBottom>
        API Settings
      </Typography>

      <TextField
        label="Etherscan API Key"
        value={etherscanKey}
        onChange={(e) => setEtherscanKey(e.target.value)}
        fullWidth
        margin="normal"
      />

      <Button
        variant="contained"
        onClick={() => console.log(etherscanKey)}
        style={{ marginTop: "10px" }}
      >
        Save
      </Button>
    </div>
  );
}
