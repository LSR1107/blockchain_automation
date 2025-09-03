import { Button, MenuItem, Select, Typography } from "@mui/material";
import { useState } from "react";

export default function SimulationDashboard() {
  const [priority, setPriority] = useState("Medium");

  return (
    <div style={{ padding: "20px" }}>
      <Typography variant="h5" gutterBottom>
        Transaction Simulation
      </Typography>
      <Typography>
        Select parameters and run a mock simulation.
      </Typography>

      <div style={{ marginTop: "20px" }}>
        <Select
          value={priority}
          onChange={(e) => setPriority(e.target.value)}
          style={{ minWidth: "150px" }}
        >
          <MenuItem value="Low">Low Priority</MenuItem>
          <MenuItem value="Medium">Medium Priority</MenuItem>
          <MenuItem value="High">High Priority</MenuItem>
        </Select>

        <Button
          variant="contained"
          style={{ marginLeft: "10px" }}
        >
          Run Simulation
        </Button>
      </div>
    </div>
  );
}
