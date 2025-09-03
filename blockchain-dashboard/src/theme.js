// src/theme.js
import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark", // switch to "light" if you prefer
    primary: { main: "#1976d2" }, // blue
    secondary: { main: "#82ca9d" }, // green
    background: {
      default: "#121212",
      paper: "#1e1e1e",
    },
    text: {
      primary: "#fff",
      secondary: "#aaa",
    },
  },
  typography: {
    fontFamily: "Inter, Roboto, sans-serif",
    h4: { fontWeight: 600 },
    h6: { fontWeight: 500 },
    body1: { fontSize: "0.95rem" },
  },
});

export default theme;
