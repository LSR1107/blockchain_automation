import { Drawer, List, ListItemButton, ListItemText } from "@mui/material";
import { Link, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  const menuItems = [
    { text: "Home", path: "/" },
    { text: "Analysis Dashboard", path: "/analysis" },
    { text: "Simulation Dashboard", path: "/simulation" },
    { text: "API Settings", path: "/api-settings" },
    { text: "About Project", path: "/about" },
  ];

  return (
    <Drawer
      variant="permanent"
      anchor="left"
      sx={{
        width: 240,
        flexShrink: 0,
        "& .MuiDrawer-paper": {
          width: 240,
          boxSizing: "border-box",
          backgroundColor: "#1E1E1E",
          color: "#fff",
        },
      }}
    >
      <List>
        {menuItems.map((item) => (
          <ListItemButton
            key={item.text}
            component={Link}
            to={item.path}
            selected={location.pathname === item.path}
            sx={{
              "&:hover": { backgroundColor: "#333" },
              "&.Mui-selected": { backgroundColor: "#1976d2", color: "#fff" },
            }}
          >
            <ListItemText primary={item.text} />
          </ListItemButton>
        ))}
      </List>
    </Drawer>
  );
}
