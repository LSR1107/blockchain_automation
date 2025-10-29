"""
multi_chain_snapshot_agent.py

Train one actor-critic to decide "wait / send on chain i / increase fee on chain i"
using independent per-chain trajectories (no global alignment required).

State = [chain0_emb || chain1_emb || ... || chain0_nextgas || chain1_nextgas || ...]
Each episode: sample an index for each chain independently (or you can set policy to use last index).
"""

import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Helper: load arrays (None if missing)
# ---------------------------
def try_load(path):
    return np.load(path) if path and os.path.exists(path) else None

# ---------------------------
# Snapshot Multi-Chain Environment
# ---------------------------
class SnapshotMultiChainEnv:
    """
    Each episode: sample an index for each chain (idx_c).
    Build a state composed of embeddings at idx_c for all chains and next_gas at idx_c for all chains.
    Action space: 0=wait, 1..C=send_now on chain (1 => chain 0), C+1..2C=increase_fee on chain
    Reward: computed for the chosen chain using its next_gas[idx_c].
    """
    def __init__(self,
                 chain_embeddings: List[np.ndarray],
                 chain_next_gas: List[np.ndarray],
                 baseline_fee_per_chain: List[float]=None,
                 fee_multiplier: float=1.5,
                 delay_penalty: float=0.2,
                 success_bonus: float=6.0,
                 success_scale: float=1.0,
                 sample_mode: str="random"):  # "random" or "last" or "sequential"
        assert len(chain_embeddings) == len(chain_next_gas)
        self.C = len(chain_embeddings)
        self.chain_embeddings = [np.asarray(e) for e in chain_embeddings]
        self.chain_next_gas = [np.asarray(g) for g in chain_next_gas]
        # lengths per chain
        self.lengths = [e.shape[0] for e in self.chain_embeddings]
        self.dim_per_chain = [e.shape[1] for e in self.chain_embeddings]
        self.total_emb_dim = sum(self.dim_per_chain)
        # baseline fees
        if baseline_fee_per_chain is None:
            self.baseline_fee = [1.0 for _ in range(self.C)]
        else:
            assert len(baseline_fee_per_chain) == self.C
            self.baseline_fee = baseline_fee_per_chain
        self.fee_multiplier = fee_multiplier
        self.delay_penalty = delay_penalty
        self.success_bonus = success_bonus
        self.success_scale = success_scale
        self.sample_mode = sample_mode
        # Keep last sampled indexes if needed
        self.current_idxs = [0] * self.C
        self.delay = 0

    def reset(self):
        """Sample new indices for each chain and return initial state"""
        self.current_idxs = []
        for i, L in enumerate(self.lengths):
            if self.sample_mode == "random":
                idx = np.random.randint(0, L)
            elif self.sample_mode == "last":
                idx = L - 1
            else:  # sequential or fallback
                idx = np.random.randint(0, L)
            self.current_idxs.append(int(idx))
        self.delay = 0
        return self._state_from_idxs(self.current_idxs)

    def _state_from_idxs(self, idxs: List[int]) -> np.ndarray:
        # gather embeddings per chain and gas per chain
        emb_parts = [self.chain_embeddings[c][idxs[c]] for c in range(self.C)]
        gas_parts = [np.asarray(self.chain_next_gas[c][idxs[c]], dtype=np.float32) for c in range(self.C)]
        state_emb = np.concatenate(emb_parts, axis=0).astype(np.float32)
        gas_vec = np.stack(gas_parts).astype(np.float32)  # shape (C,)
        state = np.concatenate([state_emb, gas_vec], axis=0)
        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        assert 0 <= action < (1 + 2*self.C)
        done = True  # snapshot episodes are one-step by default (unless action==wait we may allow continue)
        info = {}
        # fetch chosen chain and whether inc
        if action == 0:
            # wait: small negative reward and not done (optionally)
            self.delay += 1
            reward = - self.delay_penalty
            done = False  # allow agent to choose again, but beware infinite loops — we'll cap iterations externally
            info = {"action": "wait"}
        else:
            if 1 <= action <= self.C:
                chain_idx = action - 1
                inc = False
            else:
                chain_idx = action - 1 - self.C
                inc = True
            baseline = self.baseline_fee[chain_idx]
            fee = baseline * (self.fee_multiplier if inc else 1.0)
            # predicted/true next gas for chosen chain at its sampled index
            idx = self.current_idxs[chain_idx]
            predicted_gas = float(self.chain_next_gas[chain_idx][idx])
            p = self._success_prob(fee_paid=fee, predicted_gas=predicted_gas)
            reward = self.success_bonus * p - fee - self.delay_penalty * self.delay
            info = {"action": ("increase_fee" if inc else "send_now"),
                    "chain": chain_idx, "fee": fee, "success_prob": float(p)}
            done = True
        # return next_state: we resample new indices if done or keep same if not done
        if done:
            # immediate termination — if training logic expects next state, we return new sampled one
            next_state = self.reset()
        else:
            # not done -> sample a new state with same mechanism (or keep same idxs and allow repeated waits)
            # here we sample fresh indexes to diversify
            next_state = self.reset()
        return next_state, float(reward), done, info

    def _success_prob(self, fee_paid: float, predicted_gas: float) -> float:
        x = (fee_paid - predicted_gas) / self.success_scale
        p = 1.0 / (1.0 + np.exp(-x))
        return float(np.clip(p, 0.0, 1.0))


# ---------------------------
# Actor-Critic network (same as before)
# ---------------------------
class FusionActorCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128, n_actions: int = 3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
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
        value = self.critic(h).squeeze(-1)
        probs = F.softmax(logits, dim=-1)
        return probs, value


# ---------------------------
# Training loop (A2C one-step)
# ---------------------------
def train_actor_critic(env: SnapshotMultiChainEnv,
                       model: FusionActorCritic,
                       n_episodes: int = 2000,
                       gamma: float = 0.99,
                       lr: float = 3e-4,
                       device: str = "cpu",
                       max_steps_per_episode: int = 5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    rewards_hist = []
    for ep in range(1, n_episodes + 1):
        state_np = env.reset()
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
        done = False
        ep_reward = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            probs, value = model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()
            logp = dist.log_prob(torch.tensor(action, device=device))
            next_state_np, reward, done, _ = env.step(action)
            ep_reward += reward

            next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, next_value = model(next_state)

            td_target = reward + gamma * (0.0 if done else next_value.item())
            advantage = td_target - value.item()

            actor_loss = - logp * advantage
            critic_loss = 0.5 * (advantage ** 2)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            steps += 1

        rewards_hist.append(ep_reward)
        if ep % 50 == 0:
            avg = float(np.mean(rewards_hist[-200:])) if len(rewards_hist) >= 200 else float(np.mean(rewards_hist))
            print(f"Ep {ep}/{n_episodes}  avg_reward={avg:.4f}")

    return model, rewards_hist


# ---------------------------
# Demo runner (load your data or generate dummy)
# ---------------------------
if __name__ == "__main__":
    # Example file names (adjust to your saved files)
    eth_alstm = try_load("val_embeddings.npy")
    eth_y = try_load("y_val.npy")
    eth_gnn = try_load("ETH_GNN_val_embeddings.npy")
    eth_gnn_y = try_load("ETH_GNN_y_val_true.npy")

    btc_alstm = try_load("btc_val_embeddings.npy")
    btc_y = try_load("btc_y_val.npy")
    btc_gnn = try_load("Analysis_backend/saved_outputs/val_embeddings.npy")
    btc_gnn_y = try_load("Analysis_backend/saved_outputs/y_val_true.npy")

    # Build per-chain fused embeddings (concatenate ALSTM + GNN if both exist)
    chain_embs = []
    chain_gas = []
    # ETH
    if eth_alstm is not None or eth_gnn is not None:
        parts = []
        if eth_alstm is not None:
            parts.append(eth_alstm)
        if eth_gnn is not None:
            parts.append(eth_gnn)
        fused_eth = np.concatenate(parts, axis=1)
        chain_embs.append(fused_eth)
        if eth_y is not None:
            chain_gas.append(eth_y)
        elif eth_gnn_y is not None:
            chain_gas.append(eth_gnn_y)
        else:
            chain_gas.append(np.zeros(fused_eth.shape[0], dtype=np.float32))

    # BTC
    if btc_alstm is not None or btc_gnn is not None:
        parts = []
        if btc_alstm is not None:
            parts.append(btc_alstm)
        if btc_gnn is not None:
            parts.append(btc_gnn)
        fused_btc = np.concatenate(parts, axis=1)
        chain_embs.append(fused_btc)
        if btc_y is not None:
            chain_gas.append(btc_y)
        elif btc_gnn_y is not None:
            chain_gas.append(btc_gnn_y)
        else:
            chain_gas.append(np.zeros(fused_btc.shape[0], dtype=np.float32))

    # If nothing loaded, create dummy two-chain data
    if len(chain_embs) == 0:
        print("No embeddings found, creating dummy data.")
        T1, T2 = 2000, 200
        chain_embs = [np.random.randn(T1, 16).astype(np.float32), np.random.randn(T2, 12).astype(np.float32)]
        chain_gas = [(np.sin(np.arange(T1)/50.0) + 2.5).astype(np.float32),
                     (np.cos(np.arange(T2)/40.0) + 2.0).astype(np.float32)]

    # Build env model
    baseline_fees = [1.0 for _ in chain_embs]
    env = SnapshotMultiChainEnv(chain_embeddings=chain_embs,
                                chain_next_gas=chain_gas,
                                baseline_fee_per_chain=baseline_fees,
                                fee_multiplier=1.6,
                                delay_penalty=0.2,
                                success_bonus=6.0,
                                success_scale=1.0,
                                sample_mode="random")

    # Create model and train
    state_dim = env.total_emb_dim + env.C
    n_actions = 1 + 2*env.C
    model = FusionActorCritic(state_dim=state_dim, hidden_dim=128, n_actions=n_actions)

    trained_model, rewards = train_actor_critic(env, model, n_episodes=800, gamma=0.95, lr=3e-4, device="cpu")
    torch.save(trained_model.state_dict(), "multi_chain_snapshot_agent.pth")
    print("Done. Saved multi_chain_snapshot_agent.pth")
