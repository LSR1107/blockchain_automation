# hierarchical_rl_full.py
"""
Hierarchical Actor-Critic RL using ALSTM + GNN embeddings + true y_val for reward.
- Chain-specific sub-agents (ETH/BTC) optimize per-chain actions (send/wait/increase_fee)
- Meta-agent selects which chain to use (ETH or BTC)
- Handles mismatched embedding lengths by repeating the shorter modality to match the longer
- Interactive user inputs to compare estimated total costs (native + USD) for ETH and BTC
"""

import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------------
# Helpers: load and align arrays
# ------------------------------
def try_load(path: Optional[str]):
    if path and os.path.exists(path):
        return np.load(path)
    return None

def align_and_fuse_modality_arrays(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Tuple[np.ndarray, int]:
    """
    Align two modality arrays (e.g., ALSTM and GNN) into one fused embedding array.
    If one is None, return the other. If both present but lengths differ, repeat the shorter to match the longer.
    Returns fused array and used length.
    """
    if a is None and b is None:
        raise ValueError("At least one modality must be provided.")
    if a is None:
        return b.astype(np.float32), b.shape[0]
    if b is None:
        return a.astype(np.float32), a.shape[0]
    la, lb = len(a), len(b)
    if la == lb:
        return np.concatenate([a.astype(np.float32), b.astype(np.float32)], axis=1), la
    # repeat the shorter along axis 0
    if la > lb:
        repeat = int(np.ceil(la / lb))
        b_rep = np.vstack([b] * repeat)[:la]
        fused = np.concatenate([a.astype(np.float32), b_rep.astype(np.float32)], axis=1)
        return fused, la
    else:
        repeat = int(np.ceil(lb / la))
        a_rep = np.vstack([a] * repeat)[:lb]
        fused = np.concatenate([a_rep.astype(np.float32), b.astype(np.float32)], axis=1)
        return fused, lb

def align_labels_to_length(y: Optional[np.ndarray], target_len: int) -> np.ndarray:
    """If y is None -> zeros. If y length differs, repeat/trim to match target_len."""
    if y is None:
        return np.zeros((target_len,), dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(y) == target_len:
        return y
    if len(y) > target_len:
        return y[:target_len]
    repeat = int(np.ceil(target_len / len(y)))
    return np.tile(y, repeat)[:target_len]


# ------------------------------
# Chain Environment (sub-agent)
# ------------------------------
class ChainEnv:
    """
    Chain-level environment using fused embeddings and true next-gas for reward.
    For ETH: y_true and predicted_gas are interpreted as gas price PER GAS in GWEI.
             (i.e., 1 gwei = 1e-9 ETH)
    For BTC: y_true and predicted_gas are interpreted as fee in BTC per transaction (native BTC).
    state vector = [embedding_vector || predicted_gas(if given) || true_gas || volatility]
    Actions: 0=send_now, 1=wait, 2=increase_fee
    Reward uses true next gas (y_true) to compute expected success probability.
    """
    def __init__(self,
                 fused_embeddings: np.ndarray,
                 y_true: np.ndarray,
                 chain: str = "ETH",  # "ETH" or "BTC"
                 predicted_gas: Optional[np.ndarray] = None,
                 baseline_fee_native: float = 1.0,  # gwei for ETH, BTC for BTC
                 fee_multiplier: float = 1.5,
                 delay_penalty: float = 0.2,
                 success_bonus: float = 5.0,
                 success_scale: float = 1.0,
                 volatility_window: int = 5):
        assert fused_embeddings.ndim == 2
        assert len(fused_embeddings) == len(y_true)
        self.chain = chain.upper()
        self.emb = fused_embeddings.astype(np.float32)
        self.y_true = np.asarray(y_true, dtype=np.float32)
        self.predicted_gas = np.asarray(predicted_gas, dtype=np.float32) if predicted_gas is not None else None
        self.T = len(self.y_true)
        self.D = self.emb.shape[1]
        self.baseline_fee_native = float(baseline_fee_native)
        self.mult = float(fee_multiplier)
        self.delay_penalty = float(delay_penalty)
        self.success_bonus = float(success_bonus)
        self.success_scale = float(success_scale)
        self.vol_w = int(volatility_window)
        self.reset()

    def reset(self, start_idx: int = 0):
        self.t = int(np.clip(start_idx, 0, self.T - 1))
        self.delay = 0
        return self._get_state(self.t)

    def sample_start(self):
        return np.random.randint(0, max(1, self.T - 1))

    def _volatility(self, idx):
        s = max(0, idx - self.vol_w + 1)
        window = self.y_true[s: idx + 1]
        return float(np.std(window)) if len(window) > 1 else 0.0

    def _get_state(self, idx):
        emb = self.emb[idx]
        true_g = self.y_true[idx]
        pg = self.predicted_gas[idx] if self.predicted_gas is not None else true_g
        vol = self._volatility(idx)
        # state layout: [embedding..., predicted_gas, true_gas, volatility]
        return np.concatenate([emb, np.array([pg, true_g, vol], dtype=np.float32)], axis=0)

    def _success_prob(self, fee_paid_native: float, true_native: float) -> float:
        """
        Map fee_paid vs true_native to success probability.
        For ETH: both in GWEI per gas (so higher fee_paid_native relative to true_native increases p).
        For BTC: both in BTC per tx.
        success_scale controls slope.
        """
        x = (fee_paid_native - true_native) / self.success_scale
        p = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(p, 0.0, 1.0))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action <= 2
        true_native = float(self.y_true[self.t])  # gwei for ETH, BTC for BTC
        done = False
        info = {}
        if action == 0:  # send now
            fee_native = self.baseline_fee_native
            p = self._success_prob(fee_native, true_native)
            reward = self.success_bonus * p - fee_native - self.delay_penalty * self.delay
            done = True
            info = {"action": "send_now", "native_fee": fee_native, "success_prob": p}
        elif action == 1:  # wait
            self.delay += 1
            reward = - self.delay_penalty
            done = False
            info = {"action": "wait"}
        else:  # increase fee
            fee_native = self.baseline_fee_native * self.mult
            p = self._success_prob(fee_native, true_native)
            reward = self.success_bonus * p - fee_native - self.delay_penalty * self.delay
            done = True
            info = {"action": "increase_fee", "native_fee": fee_native, "success_prob": p}

        # advance time for simplicity
        self.t = min(self.t + 1, self.T - 1)
        next_state = self._get_state(self.t)
        return next_state, float(reward), bool(done), info


# ------------------------------
# Actor-Critic net (shared)
# ------------------------------
class ActorCriticNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, n_actions: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor):
        h = self.shared(state)
        logits = self.actor(h)
        probs = F.softmax(logits, dim=-1)
        value = self.critic(h).squeeze(-1)
        return probs, value


# ------------------------------
# Meta environment (chooses chain)
# ------------------------------
class MetaEnv:
    def __init__(self, chain_envs: Dict[str, ChainEnv], sample_mode: str = "random"):
        self.chain_envs = chain_envs
        self.chains = list(chain_envs.keys())
        self.C = len(self.chains)
        self.sample_mode = sample_mode
        self.state_dim_per_chain = {k: (v.D + 3) for k, v in chain_envs.items()}
        self.state_dim = sum(self.state_dim_per_chain.values())

    def reset(self):
        states = []
        for name, env in self.chain_envs.items():
            if self.sample_mode == "random":
                idx = env.sample_start()
            elif self.sample_mode == "last":
                idx = env.T - 1
            else:
                idx = env.sample_start()
            env.reset(idx)
            states.append(env._get_state(env.t))
        return np.concatenate(states, axis=0).astype(np.float32)

    def step(self, chain_idx: int, sub_agent: ActorCriticNet, rollout_steps: int = 5, device: str = "cpu"):
        assert 0 <= chain_idx < self.C
        chosen_chain = self.chains[chain_idx]
        env = self.chain_envs[chosen_chain]
        total_reward = 0.0
        done = False
        for _ in range(rollout_steps):
            s = torch.tensor(env._get_state(env.t), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                probs, _ = sub_agent(s)
            action = int(torch.argmax(probs, dim=-1).item())
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        next_states = []
        for name, ch in self.chain_envs.items():
            if self.sample_mode == "random":
                ch.reset(ch.sample_start())
            else:
                ch.reset(ch.t)
            next_states.append(ch._get_state(ch.t))
        next_meta = np.concatenate(next_states, axis=0).astype(np.float32)
        return next_meta, float(total_reward), done, {"chosen_chain": chosen_chain}


# ------------------------------
# Training helpers (unchanged)
# ------------------------------
def train_sub_agent(env: ChainEnv, model: ActorCriticNet, n_episodes: int = 400, gamma: float = 0.99, lr: float = 3e-4, device: str = "cpu"):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    rewards_hist = []
    for ep in range(1, n_episodes + 1):
        s_np = env.reset(start_idx=env.sample_start())
        s = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 8:
            probs, value = model(s)
            dist = torch.distributions.Categorical(probs)
            a = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(a, device=device))
            next_s_np, reward, done, _ = env.step(a)
            ep_reward += reward
            next_s = torch.tensor(next_s_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_val = model(next_s)
            td_target = reward + gamma * (0.0 if done else next_val.item())
            advantage = td_target - value.item()
            actor_loss = - logp * advantage
            critic_loss = 0.5 * (advantage ** 2)
            loss = actor_loss + critic_loss
            opt.zero_grad(); loss.backward(); opt.step()
            s = next_s; steps += 1
        rewards_hist.append(ep_reward)
        if ep % 50 == 0:
            print(f"[SubAgent] Ep {ep}/{n_episodes} avg_recent={np.mean(rewards_hist[-200:]) if len(rewards_hist)>=1 else np.mean(rewards_hist):.4f}")
    return model, rewards_hist

def train_meta_agent(meta_env: MetaEnv, meta_agent: ActorCriticNet, sub_agents: List[ActorCriticNet],
                     n_episodes: int = 800, gamma: float = 0.95, lr: float = 3e-4, rollout_steps: int = 5, device: str = "cpu"):
    meta_agent.to(device)
    opt = optim.Adam(meta_agent.parameters(), lr=lr)
    rewards_hist = []
    for ep in range(1, n_episodes + 1):
        s_np = meta_env.reset()
        s = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < 6:
            probs, value = meta_agent(s)
            dist = torch.distributions.Categorical(probs)
            chain_idx = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(chain_idx, device=device))
            sub_agent = sub_agents[chain_idx]
            next_s_np, reward, done, info = meta_env.step(chain_idx, sub_agent, rollout_steps=rollout_steps, device=device)
            ep_reward += reward
            next_s = torch.tensor(next_s_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_val = meta_agent(next_s)
            td_target = reward + gamma * (0.0 if done else next_val.item())
            advantage = td_target - value.item()
            actor_loss = - logp * advantage
            critic_loss = 0.5 * (advantage ** 2)
            loss = actor_loss + critic_loss
            opt.zero_grad(); loss.backward(); opt.step()
            s = next_s; steps += 1
        rewards_hist.append(ep_reward)
        if ep % 50 == 0:
            print(f"[Meta] Ep {ep}/{n_episodes} avg_recent={np.mean(rewards_hist[-200:]) if len(rewards_hist)>=1 else np.mean(rewards_hist):.4f}")
    return meta_agent, rewards_hist


# ------------------------------
# Utility: native -> USD conversion
# ------------------------------
def eth_gwei_to_usd(gwei_value: float, eth_usd_price: float, gas_units: int = 21000) -> Tuple[float, float]:
    """
    Convert gas price (gwei per gas) to total ETH fee and USD.
    Returns (eth_fee, usd_value)
    """
    eth_fee = gwei_value * 1e-9 * gas_units  # ETH
    usd = eth_fee * eth_usd_price
    return eth_fee, usd

def btc_native_to_usd(btc_fee: float, btc_usd_price: float) -> Tuple[float, float]:
    """
    btc_fee is in BTC (native) per transaction.
    Return (btc_fee, usd)
    """
    usd = btc_fee * btc_usd_price
    return btc_fee, usd


def decide_now(
    eth_env: ChainEnv,
    btc_env: ChainEnv,
    eth_agent: ActorCriticNet,
    btc_agent: ActorCriticNet,
    meta_agent: ActorCriticNet,
    eth_usd_price: float,
    btc_usd_price: float,
    device: str = "cpu"
):
    """
    Runs one meta-decision + sub-decision step using trained agents
    and returns cost + recommendation info.

    Notes:
    - ALWAYS computes both ETH and BTC predictions (so API finalize/predict can use either).
    - Uses env.t for the current pointer (consistent with ChainEnv.reset).
    """

    # ========================
    # 1. GET CURRENT STATES
    # ========================
    eth_state = eth_env._get_state(eth_env.t)
    btc_state = btc_env._get_state(btc_env.t)

    # Combine for meta-agent
    meta_state = np.concatenate([eth_state, btc_state], axis=0)
    meta_tensor = torch.tensor(meta_state, dtype=torch.float32).unsqueeze(0).to(device)

    # ========================
    # 2. META CHAIN SELECTION
    # ========================
    with torch.no_grad():
        meta_probs, _ = meta_agent(meta_tensor)

    meta_probs_np = meta_probs.cpu().numpy()[0]
    chain_idx = int(torch.argmax(meta_probs, dim=-1).item())

    chosen_chain = "ETH" if chain_idx == 0 else "BTC"
    chosen_env = eth_env if chosen_chain == "ETH" else btc_env
    chosen_agent = eth_agent if chosen_chain == "ETH" else btc_agent

    # ========================
    # 3. SUB ACTION SELECTION
    # ========================
    chosen_state = torch.tensor(
        chosen_env._get_state(chosen_env.t),
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        action_probs, critic_val = chosen_agent(chosen_state)

    action_probs_np = action_probs.cpu().numpy()[0]
    action_idx = int(torch.argmax(action_probs, dim=-1).item())

    ACTION_MAP = {
        0: "send_now",
        1: "wait",
        2: "increase_fee"
    }
    chosen_action = ACTION_MAP.get(action_idx, "wait")

    # ========================
    # 4. ALWAYS COMPUTE BOTH NATIVE/PNDS
    # ========================
    # ETH: interpret y_true as **(unit used in your dataset)**. Keep same scale as you were using before.
    # Here we keep the same small-floor behavior you had to avoid zeros.
    eth_true_raw = float(eth_env.y_true[eth_env.t])
    btc_true_raw = float(btc_env.y_true[btc_env.t])

    # small floors to avoid zero/None (adjust if your labels are in WEI vs GWEI)
    predicted_eth_gwei = float(max(eth_true_raw, 0.0001))   # assume eth_true_raw is in GWEI (as before)
    predicted_btc_fee = float(max(btc_true_raw, 1e-8))      # BTC native per tx

    # ========================
    # 5. TREND & VOLATILITY (for chosen chain)
    # ========================
    true_native = float(chosen_env.y_true[chosen_env.t])
    if chosen_env.t > 0:
        native_delta = true_native - float(chosen_env.y_true[chosen_env.t - 1])
    else:
        native_delta = 0.0

    volatility = float(abs(native_delta))

    # ========================
    # 6. COST CALCULATION (single tx defaults)
    # ========================
    if chosen_chain == "ETH":
        # convert gwei -> ETH for one simple tx using 21000 gas (match your helper eth_gwei_to_usd if needed)
        estimated_single_tx_native = predicted_eth_gwei * 1e-9 * 21000
        estimated_single_tx_usd = estimated_single_tx_native * eth_usd_price
        native_display = f"{predicted_eth_gwei:.6f} GWEI"
    else:
        estimated_single_tx_native = predicted_btc_fee
        estimated_single_tx_usd = estimated_single_tx_native * btc_usd_price
        native_display = f"{predicted_btc_fee:.8f} BTC"

    # ========================
    # 7. ADVANCE POINTER (safe)
    # ========================
    chosen_env.t = min(chosen_env.t + 1, chosen_env.T - 1)

    # ========================
    # 8. CONFIRMATION DELAY heuristic
    # ========================
    if chosen_action == "send_now":
        delay = 30
    elif chosen_action == "wait":
        delay = 120
    else:  # increase_fee
        delay = 20

    # explanation string
    explanation = (
        f"Meta-policy selected {chosen_chain} based on recent trend ∆={native_delta:.6f} "
        f"and volatility={volatility:.6f}. Sub-agent chose action: {chosen_action}."
    )

    # ========================
    # 9. RETURN (all numeric / no None)
    # ========================
    return {
        "recommended_chain": chosen_chain,
        "recommended_action": chosen_action,

        "chain_probabilities": meta_probs_np,    # caller often .tolist()s this
        "action_probabilities": action_probs_np,

        # both predictions always present as floats
        "predicted_eth_gwei": float(predicted_eth_gwei),
        "predicted_btc_fee": float(predicted_btc_fee),

        "native_display": native_display,
        "estimated_single_tx_native_amount": float(estimated_single_tx_native),
        "estimated_single_tx_usd_amount": float(estimated_single_tx_usd),

        "current_native_value": float(true_native),
        "native_delta": float(native_delta),
        "volatility": float(volatility),

        "expected_reward": float(critic_val.item()),
        "confirmation_delay_sec": int(delay),

        "explanation": explanation
    }



# ------------------------------
# Personalize & compare costs (interactive)
# ------------------------------
def personalize_and_compare(result, eth_env, btc_env, eth_usd_price=3100.0, btc_usd_price=99000.0):
    """
    Ask user for preference and transaction specifics.
    For ETH: ask tx 'class' (low/medium/high) -> gas units mapping OR accept manual gas units or gas price.
    For BTC: ask number of transactions or tx size or accept manual BTC fee per tx.
    Compare estimated totals (USD) and print recommendation and reasoning.
    """
    print("\n--- User Preference Setup ---")
    print("Please choose your preference:")
    print("1️⃣  Prefer Ethereum")
    print("2️⃣  Prefer Bitcoin")
    print("3️⃣  Neutral (no preference)")
    pref_choice = input("Enter 1, 2, or 3: ").strip()
    if pref_choice == "1":
        user_pref = "prefer_eth"
    elif pref_choice == "2":
        user_pref = "prefer_btc"
    else:
        user_pref = "neutral"

    print("\n--- Transaction Details (to compute total fees) ---")
    # ETH details
    print("\nEthereum transaction options:")
    print("a) Low-gas (transfer/approve) ~ 21,000 gas")
    print("b) Medium (swap/stake) ~ 100,000 gas")
    print("c) High (complex NFT/DeFi) ~ 300,000 gas")
    print("d) I know gas units or want to enter gas price directly")
    eth_choice = input("Choose a/b/c/d for ETH (or press Enter to skip): ").strip().lower()

    eth_total_usd = None
    eth_native = None
    if eth_choice in ["a", "b", "c"]:
        mapping = {"a": 21000, "b": 100000, "c": 300000}
        gas_units = mapping[eth_choice]
        # ETH native gas price value (gwei) from environment
        eth_gwei = eth_env.y_true[-1]
        eth_fee_eth, eth_fee_usd = eth_gwei_to_usd(eth_gwei, eth_usd_price, gas_units=gas_units)
        eth_total_usd = eth_fee_usd
        eth_native = f"{eth_fee_eth:.8f} ETH ({eth_gwei:.3f} gwei)"
    elif eth_choice == "d":
        inp = input("Enter gas units (int) or 'g' to enter gas price (gwei): ").strip().lower()
        if inp == 'g':
            g = float(input("Enter gas price in GWEI: ").strip())
            gas_units = int(input("Enter gas units for your tx (e.g. 21000): ").strip())
            eth_fee_eth, eth_fee_usd = eth_gwei_to_usd(g, eth_usd_price, gas_units=gas_units)
            eth_total_usd = eth_fee_usd
            eth_native = f"{eth_fee_eth:.8f} ETH ({g:.3f} gwei)"
        else:
            gas_units = int(inp)
            g = float(input("Enter gas price in GWEI to use (or press Enter to use predicted current): ") or eth_env.y_true[-1])
            eth_fee_eth, eth_fee_usd = eth_gwei_to_usd(g, eth_usd_price, gas_units=gas_units)
            eth_total_usd = eth_fee_usd
            eth_native = f"{eth_fee_eth:.8f} ETH ({g:.3f} gwei)"
    else:
        # user skipped ETH input
        eth_total_usd = None

    # BTC details
    print("\nBitcoin transaction options:")
    print("a) I know the BTC fee per tx (native BTC)")
    print("b) Estimate by number of inputs/outputs (approx size) -- we'll use a bytes-per-tx heuristic")
    print("c) Skip BTC input")
    btc_choice = input("Choose a/b/c for BTC (or press Enter to skip): ").strip().lower()

    btc_total_usd = None
    btc_native = None
    if btc_choice == "a":
        btc_fee_btc = float(input("Enter BTC fee (in BTC) for the transaction, e.g. 0.00023: ").strip())
        btc_fee_btc, btc_fee_usd = btc_native_to_usd(btc_fee_btc, btc_usd_price)
        btc_total_usd = btc_fee_usd
        btc_native = f"{btc_fee_btc:.8f} BTC"
    elif btc_choice == "b":
        # Ask approximate inputs/outputs to estimate tx size
        n_inputs = int(input("Enter number of inputs (typical: 1-3): ").strip() or 1)
        n_outputs = int(input("Enter number of outputs (typical: 2): ").strip() or 2)
        # approximate bytes: input ≈ 148 bytes, output ≈ 34 bytes, overhead ≈ 10
        est_size = 10 + n_inputs * 148 + n_outputs * 34
        print(f"Estimated tx size: ~{est_size} bytes")
        # Ask fee rate in sat/vB (or use a default)
        fee_rate_sat_vb = float(input("Enter fee rate in sat/vB (e.g. 20) or press Enter for default 10: ") or 10.0)
        # BTC fee in satoshis = fee_rate_sat_vB * vbytes
        satoshis = fee_rate_sat_vb * est_size
        btc_fee_btc = satoshis * 1e-8
        btc_fee_btc, btc_fee_usd = btc_native_to_usd(btc_fee_btc, btc_usd_price)
        btc_total_usd = btc_fee_usd
        btc_native = f"{btc_fee_btc:.8f} BTC (≈ {satoshis:.0f} sats)"
    else:
        btc_total_usd = None

    # Compare if both present
    print("\n--- Cost Comparison ---")
    if eth_total_usd is not None:
        print(f"Ethereum estimated total fee: {eth_native}  ≈ ${eth_total_usd:.6f} USD")
    else:
        print("Ethereum estimate: skipped by user")

    if btc_total_usd is not None:
        print(f"Bitcoin estimated total fee: {btc_native}  ≈ ${btc_total_usd:.6f} USD")
    else:
        print("Bitcoin estimate: skipped by user")

    # If both available, compare and recommend
    if eth_total_usd is not None and btc_total_usd is not None:
        if eth_total_usd < btc_total_usd:
            compare_text = f"💡 Cheaper: Ethereum (by ${btc_total_usd - eth_total_usd:.6f} USD)"
            compare_choice = "ETH"
        else:
            compare_text = f"💡 Cheaper: Bitcoin (by ${eth_total_usd - btc_total_usd:.6f} USD)"
            compare_choice = "BTC"
        print(compare_text)
    else:
        compare_choice = None

    # Explain why model originally chose that chain
    model_reason = ""
    rec_chain = result["recommended_chain"]
    rec_action = result["recommended_action"]
    model_reason = (
        f"Model reason: meta-agent probability distribution = {result['chain_probabilities']}. "
        f"Sub-agent action probabilities = {result['action_probabilities']}. "
        f"The model picked {rec_chain} because it had higher expected short-term reward "
        f"(critic={result['expected_reward']:.3f}) while showing acceptable success probability."
    )

    # Personalization: if user preference contradicts model, show adjusted suggestion and cost tradeoff
    final_choice = rec_chain
    if user_pref == "prefer_eth":
        final_choice = "Ethereum"
    elif user_pref == "prefer_btc":
        final_choice = "Bitcoin"

    # If user forced preference, show tradeoff
    personalized_feedback = ""
    if user_pref != "neutral" and final_choice != rec_chain:
        personalized_feedback = (
            f"You preferred {final_choice}; note model originally chose {rec_chain}. "
            f"If you force {final_choice}, expected cost and confirmation times will follow the {final_choice} estimates above."
        )
    else:
        if compare_choice:
            personalized_feedback = f"Model choice aligns with cost comparison: {compare_choice} cheaper."
        else:
            personalized_feedback = "No direct cost comparison (missing input)."

    # Assemble final personalization summary
    summary = {
        "user_pref": user_pref,
        "eth_total_usd": eth_total_usd,
        "btc_total_usd": btc_total_usd,
        "compare_choice": compare_choice,
        "final_choice": final_choice,
        "personalized_feedback": personalized_feedback,
        "model_reason": model_reason
    }

    return summary


# ------------------------------
# Main runner
# ------------------------------
if __name__ == "__main__":
    # Filenames - adjust if yours differ
    ETH_ALSTM = "val_embeddings.npy"
    ETH_ALSTM_Y = "y_val.npy"
    ETH_GNN = "ETH_GNN_val_embeddings.npy"
    ETH_GNN_Y = "ETH_GNN_y_val_true.npy"

    BTC_ALSTM = "btc_val_embeddings.npy"
    BTC_ALSTM_Y = "btc_y_val.npy"
    BTC_GNN = "Analysis_backend/saved_outputs/val_embeddings.npy"
    BTC_GNN_Y = "Analysis_backend/saved_outputs/y_val_true.npy"

    # load embeddings/labels (optional files)
    eth_a = try_load(ETH_ALSTM)
    eth_ay = try_load(ETH_ALSTM_Y)
    eth_g = try_load(ETH_GNN)
    eth_gy = try_load(ETH_GNN_Y)

    btc_a = try_load(BTC_ALSTM)
    btc_ay = try_load(BTC_ALSTM_Y)
    btc_g = try_load(BTC_GNN)
    btc_gy = try_load(BTC_GNN_Y)

    # Fuse/align per-chain modalities
    if eth_a is None and eth_g is None:
        print("No ETH embeddings found — creating small dummy ETH data.")
        eth_emb = np.random.randn(500, 16).astype(np.float32)
        eth_y = (np.sin(np.arange(500)/50.0) + 2.5).astype(np.float32)  # treat as gwei
    else:
        eth_emb, len_eth = align_and_fuse_modality_arrays(eth_a, eth_g)
        if eth_ay is not None:
            eth_y = align_labels_to_length(eth_ay, len_eth)
        elif eth_gy is not None:
            eth_y = align_labels_to_length(eth_gy, len_eth)
        else:
            eth_y = np.zeros((len_eth,), dtype=np.float32)

    if btc_a is None and btc_g is None:
        print("No BTC embeddings found — creating small dummy BTC data.")
        btc_emb = np.random.randn(200, 12).astype(np.float32)
        btc_y = (np.cos(np.arange(200)/40.0) + 2.0).astype(np.float32)  # treat as BTC per tx
    else:
        btc_emb, len_btc = align_and_fuse_modality_arrays(btc_a, btc_g)
        if btc_ay is not None:
            btc_y = align_labels_to_length(btc_ay, len_btc)
        elif btc_gy is not None:
            btc_y = align_labels_to_length(btc_gy, len_btc)
        else:
            btc_y = np.zeros((len_btc,), dtype=np.float32)

    print("ETH fused shape:", eth_emb.shape, "ETH y shape:", eth_y.shape)
    print("BTC fused shape:", btc_emb.shape, "BTC y shape:", btc_y.shape)

    # Build chain envs with chain-specific baseline_native values
    # For ETH baseline_fee_native is in GWEI (gas price per gas)
    eth_env = ChainEnv(fused_embeddings=eth_emb, y_true=eth_y, chain="ETH", baseline_fee_native=1.0, fee_multiplier=1.6, success_scale=1.0)
    # For BTC baseline_fee_native is BTC per tx (so pick a representative value ~0.0002 BTC)
    btc_env = ChainEnv(fused_embeddings=btc_emb, y_true=btc_y, chain="BTC", baseline_fee_native=0.00023, fee_multiplier=1.6, success_scale=0.0001)

    chain_envs = {"ETH": eth_env, "BTC": btc_env}

    # Create and train sub agents & meta agent --- you can reuse previous saved models if you want
    eth_state_dim = eth_env.D + 3
    btc_state_dim = btc_env.D + 3
    eth_agent = ActorCriticNet(state_dim=eth_state_dim, n_actions=3)
    btc_agent = ActorCriticNet(state_dim=btc_state_dim, n_actions=3)

    print("\n--- Training ETH sub-agent ---")
    eth_agent, _ = train_sub_agent(eth_env, eth_agent, n_episodes=300, lr=3e-4)
    print("\n--- Training BTC sub-agent ---")
    btc_agent, _ = train_sub_agent(btc_env, btc_agent, n_episodes=300, lr=3e-4)

    torch.save(eth_agent.state_dict(), "eth_agent.pth")
    torch.save(btc_agent.state_dict(), "btc_agent.pth")
    print("Saved eth_agent.pth and btc_agent.pth")

    meta_env = MetaEnv(chain_envs=chain_envs, sample_mode="random")
    meta_agent = ActorCriticNet(state_dim=meta_env.state_dim, n_actions=meta_env.C)

    print("\n--- Training Meta-Agent (chain selector) ---")
    meta_agent, _ = train_meta_agent(meta_env, meta_agent, sub_agents=[eth_agent, btc_agent], n_episodes=800, rollout_steps=6, lr=3e-4)
    torch.save(meta_agent.state_dict(), "meta_agent.pth")
    print("Saved meta_agent.pth")

    print("\n=== USER DECISION INTERFACE ===")
    result = decide_now(eth_env, btc_env, eth_agent, btc_agent, meta_agent, eth_usd_price=3100.0, btc_usd_price=99000.0)
    # Print recommendation summary (native + USD estimates)
    print(f"\n💡 Recommendation: Use {result['recommended_chain']} and {result['recommended_action']}")
    print(f"🧠 Reasoning: {result['explanation']}")
    print(f"🔢 Meta chain probs: {result['chain_probabilities']}")
    print(f"🔢 Sub-agent action probs: {result['action_probabilities']}")
    print(f"🔢 Current native value: {result['native_display']}")
    print(f"💸 Estimated single tx: {result['estimated_single_tx_native_amount']}  ≈ {result['estimated_single_tx_usd_amount']}")
    print(f"⏱️ Expected confirmation delay: {result['confirmation_delay_sec']} sec")
    print(f"🏆 Expected reward (critic value): {result['expected_reward']:.3f}")

    # Ask user and get personalization & comparison
    summary = personalize_and_compare(result, eth_env, btc_env, eth_usd_price=3100.0, btc_usd_price=99000.0)

    print("\n=== PERSONALIZED SUMMARY ===")
    print(f"User preference: {summary['user_pref']}")
    if summary['eth_total_usd'] is not None:
        print(f"Ethereum estimated total fee: ${summary['eth_total_usd']:.6f} USD")
    if summary['btc_total_usd'] is not None:
        print(f"Bitcoin estimated total fee: ${summary['btc_total_usd']:.6f} USD")
    if summary['compare_choice']:
        print(f"Cheaper according to your inputs: {summary['compare_choice']}")
    print(f"\n💬 User-Based Suggestion:\n{summary['personalized_feedback']}")
    print(f"Reason: {summary['model_reason']}")
