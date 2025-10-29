"""
hierarchical_rl_full.py

Hierarchical Actor-Critic RL using ALSTM + GNN embeddings + true y_val for reward.
- Chain-specific sub-agents (ETH/BTC) optimize per-chain actions (send/wait/increase_fee)
- Meta-agent selects which chain to use (ETH or BTC)
- Handles mismatched embedding lengths by repeating the smaller modal to match the larger within a chain
"""

import os
from typing import List, Tuple, Dict, Optional
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
    state vector = [embedding_vector || predicted_gas(if given) || true_gas || volatility]
    Actions: 0=send_now, 1=wait, 2=increase_fee
    Reward uses true next gas (y_true) to compute actual expected success probability.
    """
    def __init__(self,
                 fused_embeddings: np.ndarray,
                 y_true: np.ndarray,
                 predicted_gas: Optional[np.ndarray] = None,
                 baseline_fee_gwei: float = 1.0,
                 fee_multiplier: float = 1.5,
                 delay_penalty: float = 0.2,
                 success_bonus: float = 5.0,
                 success_scale: float = 1.0,
                 volatility_window: int = 5):
        assert fused_embeddings.ndim == 2
        assert len(fused_embeddings) == len(y_true)
        self.emb = fused_embeddings.astype(np.float32)
        self.y_true = np.asarray(y_true, dtype=np.float32)
        self.predicted_gas = np.asarray(predicted_gas, dtype=np.float32) if predicted_gas is not None else None
        self.T = len(self.y_true)
        self.D = self.emb.shape[1]
        self.baseline_fee = float(baseline_fee_gwei)
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

    def _success_prob(self, fee_paid: float, true_gas: float) -> float:
        # Use true_gas as ground-truth comparator (higher true_gas => need higher fee)
        x = (fee_paid - true_gas) / self.success_scale
        p = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(p, 0.0, 1.0))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action <= 2
        true_g = float(self.y_true[self.t])
        done = False
        info = {}
        if action == 0:  # send now
            fee = self.baseline_fee
            p = self._success_prob(fee, true_g)
            reward = self.success_bonus * p - fee - self.delay_penalty * self.delay
            done = True
            info = {"action": "send_now", "fee": fee, "success_prob": p}
        elif action == 1:  # wait
            self.delay += 1
            reward = - self.delay_penalty  # small penalty for waiting
            done = False
            info = {"action": "wait"}
        else:  # increase fee
            fee = self.baseline_fee * self.mult
            p = self._success_prob(fee, true_g)
            reward = self.success_bonus * p - fee - self.delay_penalty * self.delay
            done = True
            info = {"action": "increase_fee", "fee": fee, "success_prob": p}

        # advance time (if done or not done, move to next sample for simplicity)
        self.t = min(self.t + 1, self.T - 1)
        next_state = self._get_state(self.t)
        return next_state, float(reward), bool(done), info


# ------------------------------
# Actor-Critic network (shared)
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
    """
    Meta-env samples one index per chain (or uses last), forms a joint meta-state,
    and allows meta-agent to choose a chain. The chosen chain's sub-agent performs a rollout and returns cumulative reward.
    """
    def __init__(self, chain_envs: Dict[str, ChainEnv], sample_mode: str = "random"):
        self.chain_envs = chain_envs
        self.chains = list(chain_envs.keys())
        self.C = len(self.chains)
        self.sample_mode = sample_mode
        # meta state dimension = sum(chain_state_dims)
        self.state_dim_per_chain = {k: (v.D + 3) for k, v in chain_envs.items()}  # emb + (pg, true_g, vol)
        self.state_dim = sum(self.state_dim_per_chain.values())

    def reset(self):
        # sample indices depending on mode; env.reset will return state vector for that chain
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
        meta_state = np.concatenate(states, axis=0).astype(np.float32)
        return meta_state

    def step(self, chain_idx: int, sub_agent: ActorCriticNet, rollout_steps: int = 5, device: str = "cpu"):
        """
        chain_idx: which chain to pick (integer)
        sub_agent: chain-specific policy network (ActorCriticNet)
        Performs a short rollout using greedy policy (or sampling) from sub_agent on chosen chain.
        Returns next_meta_state, cumulative_reward, done, info
        """
        assert 0 <= chain_idx < self.C
        chosen_chain = self.chains[chain_idx]
        env = self.chain_envs[chosen_chain]

        # short rollout (apply the sub-agent policy)
        total_reward = 0.0
        done = False
        for step in range(rollout_steps):
            s = torch.tensor(env._get_state(env.t), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                probs, _ = sub_agent(s)
            # choose greedy or sample; use greedy to reflect sub-agent policy exploitation in meta rollout
            action = int(torch.argmax(probs, dim=-1).item())
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break

        # Build next meta-state by sampling/resetting all chains
        next_states = []
        for name, ch in self.chain_envs.items():
            # for diversity, sample new index for non-chosen chains too (or keep)
            if self.sample_mode == "random":
                ch.reset(ch.sample_start())
            else:
                ch.reset(ch.t)  # keep
            next_states.append(ch._get_state(ch.t))
        next_meta = np.concatenate(next_states, axis=0).astype(np.float32)
        return next_meta, float(total_reward), done, {"chosen_chain": chosen_chain}


# ------------------------------
# Training helpers
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

def decide_now(eth_env, btc_env, eth_agent, btc_agent, meta_agent, device="cpu"):
    """Return a user-facing decision combining both levels of RL."""

    # --- 1️⃣  Get current states (latest embeddings) ---
    eth_state = eth_env._get_state(-1)
    btc_state = btc_env._get_state(-1)
    meta_state = np.concatenate([eth_state, btc_state], axis=0)
    meta_state_t = torch.tensor(meta_state, dtype=torch.float32, device=device).unsqueeze(0)

    # --- 2️⃣  Meta-agent: choose which chain ---
    with torch.no_grad():
        chain_probs, _ = meta_agent(meta_state_t)
    chain_idx = int(torch.argmax(chain_probs, dim=-1).item())
    chosen_chain = ["Ethereum", "Bitcoin"][chain_idx]

    # --- 3️⃣  Sub-agent: decide action for that chain ---
    sub_agent = eth_agent if chosen_chain == "Ethereum" else btc_agent
    env = eth_env if chosen_chain == "Ethereum" else btc_env
    s_t = torch.tensor(env._get_state(-1), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        act_probs, _ = sub_agent(s_t)
    act_idx = int(torch.argmax(act_probs, dim=-1).item())
    action_names = ["Send Now", "Wait", "Increase Fee"]
    chosen_action = action_names[act_idx]

    # --- 4️⃣  Generate user-friendly explanation ---
    current_gas = float(env.y_true[-1])
    recent_vol = np.std(env.y_true[-5:]) if len(env.y_true) > 5 else 0
    explanation = (
        f"Based on recent gas trends (current: {current_gas:.2f}, volatility: {recent_vol:.2f}), "
        f"and overall network embeddings, "
        f"the meta-policy prefers {chosen_chain} with action '{chosen_action}'."
    )

    result = {
        "recommended_chain": chosen_chain,
        "recommended_action": chosen_action,
        "chain_probabilities": chain_probs.cpu().numpy().flatten(),
        "action_probabilities": act_probs.cpu().numpy().flatten(),
        "explanation": explanation
    }

    return result
# ------------------------------
# Main runner
# ------------------------------
if __name__ == "__main__":
    # Filenames - adjust if yours differ
    # ETH files
    ETH_ALSTM = "val_embeddings.npy"         # (N1, d1)
    ETH_ALSTM_Y = "y_val.npy"                     # (N1,)
    ETH_GNN = "ETH_GNN_val_embeddings.npy"            # (N2, d2) optional
    ETH_GNN_Y = "ETH_GNN_y_val_true.npy"              # (N2,) optional

    # BTC files
    BTC_ALSTM = "btc_val_embeddings.npy"
    BTC_ALSTM_Y = "btc_y_val.npy"
    BTC_GNN = "Analysis_backend/saved_outputs/val_embeddings.npy"
    BTC_GNN_Y = "Analysis_backend/saved_outputs/y_val_true.npy"

    # load ETH
    eth_a = try_load(ETH_ALSTM)
    eth_ay = try_load(ETH_ALSTM_Y)
    eth_g = try_load(ETH_GNN)
    eth_gy = try_load(ETH_GNN_Y)

    # load BTC
    btc_a = try_load(BTC_ALSTM)
    btc_ay = try_load(BTC_ALSTM_Y)
    btc_g = try_load(BTC_GNN)
    btc_gy = try_load(BTC_GNN_Y)

    # Fuse/align per-chain modalities
    if eth_a is None and eth_g is None:
        print("No ETH embeddings found — creating small dummy ETH data.")
        eth_emb = np.random.randn(500, 16).astype(np.float32)
        eth_y = (np.sin(np.arange(500)/50.0) + 2.5).astype(np.float32)
    else:
        eth_emb, len_eth = align_and_fuse_modality_arrays(eth_a, eth_g)
        # prefer ALSTM y if available, else GNN y, else zeros
        if eth_ay is not None:
            eth_y = align_labels_to_length(eth_ay, len_eth)
        elif eth_gy is not None:
            eth_y = align_labels_to_length(eth_gy, len_eth)
        else:
            eth_y = np.zeros((len_eth,), dtype=np.float32)

    if btc_a is None and btc_g is None:
        print("No BTC embeddings found — creating small dummy BTC data.")
        btc_emb = np.random.randn(200, 12).astype(np.float32)
        btc_y = (np.cos(np.arange(200)/40.0) + 2.0).astype(np.float32)
    else:
        btc_emb, len_btc = align_and_fuse_modality_arrays(btc_a, btc_g)
        if btc_ay is not None:
            btc_y = align_labels_to_length(btc_ay, len_btc)
        elif btc_gy is not None:
            btc_y = align_labels_to_length(btc_gy, len_btc)
        else:
            btc_y = np.zeros((len_btc,), dtype=np.float32)

    # Print shapes for diagnostics
    print("ETH fused shape:", eth_emb.shape, "ETH y shape:", eth_y.shape)
    print("BTC fused shape:", btc_emb.shape, "BTC y shape:", btc_y.shape)

    # Build chain envs
    eth_env = ChainEnv(fused_embeddings=eth_emb, y_true=eth_y, baseline_fee_gwei=1.0, fee_multiplier=1.6)
    btc_env = ChainEnv(fused_embeddings=btc_emb, y_true=btc_y, baseline_fee_gwei=1.0, fee_multiplier=1.6)
    chain_envs = {"ETH": eth_env, "BTC": btc_env}

    # Init sub-agents
    eth_state_dim = eth_env.D + 3
    btc_state_dim = btc_env.D + 3
    eth_agent = ActorCriticNet(state_dim=eth_state_dim, n_actions=3)
    btc_agent = ActorCriticNet(state_dim=btc_state_dim, n_actions=3)

    # Train sub-agents
    print("\n--- Training ETH sub-agent ---")
    eth_agent, eth_hist = train_sub_agent(eth_env, eth_agent, n_episodes=300, lr=3e-4)
    print("\n--- Training BTC sub-agent ---")
    btc_agent, btc_hist = train_sub_agent(btc_env, btc_agent, n_episodes=300, lr=3e-4)

    # Save sub-agent weights
    torch.save(eth_agent.state_dict(), "eth_agent.pth")
    torch.save(btc_agent.state_dict(), "btc_agent.pth")
    print("Saved eth_agent.pth and btc_agent.pth")

    # Build meta-env and meta-agent
    meta_env = MetaEnv(chain_envs=chain_envs, sample_mode="random")
    meta_agent = ActorCriticNet(state_dim=meta_env.state_dim, n_actions=meta_env.C)

    print("\n--- Training Meta-Agent (chain selector) ---")
    meta_agent, meta_hist = train_meta_agent(meta_env, meta_agent, sub_agents=[eth_agent, btc_agent],
                                            n_episodes=800, rollout_steps=6, lr=3e-4)

    torch.save(meta_agent.state_dict(), "meta_agent.pth")
    print("Saved meta_agent.pth")

    print("\nTraining complete. Models saved: eth_agent.pth, btc_agent.pth, meta_agent.pth")

    print("\n=== USER DECISION INTERFACE ===")
    result = decide_now(eth_env, btc_env, eth_agent, btc_agent, meta_agent)
    print(f"\n💡 Recommendation: Use {result['recommended_chain']} and {result['recommended_action']}")
    print(f"🧠 Reasoning: {result['explanation']}")
    print(f"🔢 Meta chain probs: {result['chain_probabilities']}")
    print(f"🔢 Sub-agent action probs: {result['action_probabilities']}")
