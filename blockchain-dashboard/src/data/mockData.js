// Centralized mock data for blockchain dashboards

export const tpsData = [
  { time: "10:00", tps: 45 },
  { time: "10:05", tps: 50 },
  { time: "10:10", tps: 55 },
  { time: "10:15", tps: 53 },
];

export const gasData = [
  { time: "10:00", gas: 20 },
  { time: "10:05", gas: 25 },
  { time: "10:10", gas: 22 },
  { time: "10:15", gas: 18 },
];

export const blockData = [
  { time: "10:00", size: 1.2 },
  { time: "10:05", size: 1.3 },
  { time: "10:10", size: 1.5 },
  { time: "10:15", size: 1.4 },
];

export const congestionValue = 0.6;

export const recommendations = {
  action: "Execute Now",
  feeRange: "0.00021 - 0.00025 ETH",
  time: 30,
  confidence: 92,
};
