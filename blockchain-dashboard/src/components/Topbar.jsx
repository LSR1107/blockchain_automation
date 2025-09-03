import { AppBar, Toolbar, Typography, Select, MenuItem } from "@mui/material";

export default function Topbar({ selectedChain, onChangeChain }) {
  const drawerWidth = 240;

  return (
    <AppBar
      position="fixed"
      sx={{
        backgroundColor: "#1976d2",
        zIndex: (theme) => theme.zIndex.drawer + 1,
        width: `calc(100% - ${drawerWidth}px)`,
        marginLeft: `${drawerWidth}px`,
      }}
    >
      <Toolbar>
        <Typography
          variant="h6"
          sx={{ fontWeight: "bold", fontSize: "1.3rem", flexGrow: 1 }}
        >
          Blockchain Scalability Dashboard
        </Typography>

        <Select
          value={selectedChain}
          onChange={(e) => onChangeChain(e.target.value)}
          sx={{
            backgroundColor: "#1e1e1e",
            color: "#fff",
            borderRadius: "6px",
            "& .MuiSvgIcon-root": { color: "#fff" },
          }}
        >
          <MenuItem value="Ethereum">Ethereum</MenuItem>
          <MenuItem value="Bitcoin">Bitcoin</MenuItem>
          <MenuItem value="Solana">Solana</MenuItem>
          <MenuItem value="Custom">Custom</MenuItem>
        </Select>
      </Toolbar>
    </AppBar>
  );
}
