import numpy as np
from datetime import datetime

# Import your actual RL agent
from Analysis_backend.multi_chain_A2C import MultiChainA2C
from Analysis_backend.RLagent import RLAgent

def run_reinforcement_analysis():

    try:
        # Initialize
        env = MultiChainA2C()
        agent = RLAgent(env.state_size, env.action_size)

        state = env.reset()

        action, action_probs = agent.act(state)

        next_state, reward, done, info = env.step(action)

        chain = info["chain"]
        suggested_action = info["action"]

        result = {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "recommended_chain": chain,
            "suggested_action": suggested_action,
            "confidence": float(np.max(action_probs)),
            "meta_chain_probabilities": action_probs.tolist(),
            "state_vector": state.tolist()
        }

        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}
