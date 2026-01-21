---
sidebar_position: 10
---

# Reinforcement Learning with DQN

Learn how to implement Deep Q-Networks (DQN) for reinforcement learning using Flax NNX. This guide covers the foundational concepts of RL and provides a complete implementation for training agents to solve control tasks.

:::info Example Code
See the full implementation: [`examples/advanced/dqn_reinforcement_learning.py`](https://github.com/mlnomadpy/flaxdocs/blob/main/examples/advanced/dqn_reinforcement_learning.py)
:::

## What is Reinforcement Learning?

Reinforcement learning (RL) is a paradigm where an **agent** learns to make decisions by interacting with an **environment**:

```
┌─────────────────────────────────────────────────────────┐
│                 RL Interaction Loop                      │
└─────────────────────────────────────────────────────────┘

        ┌─────────────┐          action (a)
        │             │  ─────────────────────────►
        │    Agent    │                              ┌─────────────┐
        │             │  ◄─────────────────────────  │             │
        └─────────────┘    state (s), reward (r)     │ Environment │
                                                      │             │
                                                      └─────────────┘

Goal: Learn a policy π(a|s) that maximizes cumulative reward
```

**Key concepts:**
- **State (s)**: Observation of the environment
- **Action (a)**: Decision made by the agent
- **Reward (r)**: Feedback signal from the environment
- **Policy (π)**: Strategy for selecting actions
- **Value function**: Expected cumulative reward from a state

## Deep Q-Networks (DQN)

DQN combines Q-learning with deep neural networks to handle high-dimensional state spaces.

### The Q-Function

The Q-function (action-value function) estimates the expected return from taking action $a$ in state $s$:

$$
Q^*(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a\right]
$$

where $\gamma \in [0, 1]$ is the discount factor.

### Bellman Equation

The optimal Q-function satisfies the Bellman optimality equation:

$$
Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')
$$

This recursive relationship is the foundation for Q-learning updates.

### DQN Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Q-Network                             │
└─────────────────────────────────────────────────────────┘

    State s              Hidden Layers              Q-values
  ┌─────────┐          ┌─────────────┐          ┌─────────┐
  │ s₁      │          │             │          │ Q(s,a₁) │
  │ s₂      │  ────►   │   Neural    │  ────►   │ Q(s,a₂) │
  │ ...     │          │   Network   │          │ Q(s,a₃) │
  │ sₙ      │          │             │          │ ...     │
  └─────────┘          └─────────────┘          └─────────┘
```

## Implementation in Flax NNX

### Q-Network Model

```python
from flax import nnx
import jax.numpy as jnp

class QNetwork(nnx.Module):
    """
    Deep Q-Network for estimating action values.
    
    Maps state observations to Q-values for each action.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        rngs: nnx.Rngs
    ):
        """
        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer dimension
            rngs: Random number generators
        """
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, action_dim, rngs=rngs)
    
    def __call__(self, state: jax.Array) -> jax.Array:
        """
        Compute Q-values for all actions given a state.
        
        Args:
            state: State observation of shape (batch, state_dim)
            
        Returns:
            Q-values of shape (batch, action_dim)
        """
        x = self.fc1(state)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        q_values = self.fc3(x)
        return q_values
```

**Key points:**
- Input is the state vector
- Output has one Q-value per possible action
- ReLU activations in hidden layers
- No activation on output (Q-values can be any real number)

### Dueling Architecture

The Dueling DQN separates value and advantage estimation:

$$
Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|}\sum_{a'} A(s, a')\right)
$$

```python
class DuelingQNetwork(nnx.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        rngs: nnx.Rngs
    ):
        # Shared feature layer
        self.feature = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        
        # Value stream: V(s)
        self.value_fc = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.value_out = nnx.Linear(hidden_dim // 2, 1, rngs=rngs)
        
        # Advantage stream: A(s, a)
        self.advantage_fc = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.advantage_out = nnx.Linear(hidden_dim // 2, action_dim, rngs=rngs)
    
    def __call__(self, state: jax.Array) -> jax.Array:
        # Shared features
        features = nnx.relu(self.feature(state))
        
        # Value stream
        value = nnx.relu(self.value_fc(features))
        value = self.value_out(value)  # (batch, 1)
        
        # Advantage stream
        advantage = nnx.relu(self.advantage_fc(features))
        advantage = self.advantage_out(advantage)  # (batch, action_dim)
        
        # Combine: Q(s, a) = V(s) + (A(s, a) - mean(A))
        q_values = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
        
        return q_values
```

**Benefits of Dueling:**
- Better generalization across actions
- More efficient learning when many actions have similar values
- Separates "how good is this state" from "how much better is this action"

## Experience Replay

Experience replay stores transitions and samples them randomly for training:

```python
from collections import deque
from typing import NamedTuple
import random

class Transition(NamedTuple):
    """Single transition tuple (s, a, r, s', done)."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Enables:
    - Breaking correlation between consecutive samples
    - Reusing experiences multiple times
    - Stable learning with mini-batch gradient descent
    """
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, jax.Array]:
        """Sample a random batch of transitions."""
        transitions = random.sample(self.buffer, batch_size)
        
        return {
            'states': jnp.array([t.state for t in transitions]),
            'actions': jnp.array([t.action for t in transitions]),
            'rewards': jnp.array([t.reward for t in transitions]),
            'next_states': jnp.array([t.next_state for t in transitions]),
            'dones': jnp.array([t.done for t in transitions], dtype=jnp.float32)
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
```

**Why experience replay matters:**
1. **Breaks temporal correlation**: Sequential samples are highly correlated; random sampling decorrelates
2. **Data efficiency**: Each experience can be used multiple times
3. **Stability**: Diverse batches lead to more stable gradients

## Epsilon-Greedy Exploration

Balance exploration (trying new actions) and exploitation (using known good actions):

```python
class EpsilonGreedy:
    """
    Epsilon-greedy exploration strategy.
    
    With probability epsilon: take random action (explore)
    With probability 1-epsilon: take best action (exploit)
    """
    
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000
    ):
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0
    
    @property
    def epsilon(self) -> float:
        """Current epsilon value with linear decay."""
        progress = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def select_action(self, q_values: jax.Array, num_actions: int) -> int:
        """Select action using epsilon-greedy policy."""
        self.step_count += 1
        
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, num_actions - 1)
        else:
            # Exploit: best action
            return int(jnp.argmax(q_values))
```

**Exploration schedule:**
```
Epsilon
  1.0  ─────┐
            │╲
            │ ╲
            │  ╲
            │   ╲
  0.01 ─────│────────────────────────
            │
            └────────────────────────► Steps
                 Decay period
```

## Target Network

Use a separate target network for stable Q-value targets:

```python
def _soft_update_target(self):
    """Soft update target network: θ_target = τ*θ + (1-τ)*θ_target."""
    q_params = nnx.state(self.q_network, nnx.Param)
    target_params = nnx.state(self.target_network, nnx.Param)
    
    new_target_params = jax.tree.map(
        lambda q, t: self.tau * q + (1 - self.tau) * t,
        q_params,
        target_params
    )
    nnx.update(self.target_network, new_target_params)
```

**Why target networks help:**
- Q-learning targets use the same network being updated → instability
- Target network provides stable targets
- Soft updates (τ ≈ 0.005) provide smooth transitions

## Training Step

The core DQN training step:

```python
@nnx.jit
def _update(self, batch: Dict[str, jax.Array]) -> jax.Array:
    """
    Compute TD loss and update Q-network.
    
    Uses the Bellman equation:
    Q(s, a) ← r + γ * max_a' Q_target(s', a')
    """
    def loss_fn(model):
        # Current Q-values for taken actions
        q_values = model(batch['states'])
        q_values_selected = q_values[
            jnp.arange(len(batch['actions'])),
            batch['actions']
        ]
        
        # Target Q-values (no gradient through target network)
        next_q_values = self.target_network(batch['next_states'])
        next_q_max = jnp.max(next_q_values, axis=-1)
        
        # TD target: r + γ * max Q(s', a') * (1 - done)
        targets = batch['rewards'] + self.gamma * next_q_max * (1 - batch['dones'])
        
        # Mean squared TD error
        td_errors = q_values_selected - targets
        loss = jnp.mean(td_errors ** 2)
        
        return loss
    
    # Compute gradients and update
    loss, grads = nnx.value_and_grad(loss_fn)(self.q_network)
    self.optimizer.update(self.q_network, grads)
    
    return loss
```

**TD Error visualization:**

```
                     Predicted Q(s,a)
                           ↓
TD Error = Q(s,a) - [r + γ * max Q(s',a')]
                           ↑
                     TD Target (from target network)
```

## Complete DQN Agent

```python
class DQNAgent:
    """
    Deep Q-Network Agent combining all components.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        use_dueling: bool = False,
        seed: int = 0
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Initialize networks
        rngs = nnx.Rngs(seed)
        
        NetworkClass = DuelingQNetwork if use_dueling else QNetwork
        self.q_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rngs=rngs
        )
        
        # Create target network
        self.target_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rngs=nnx.Rngs(seed + 1)
        )
        self._hard_update_target()
        
        # Optimizer
        self.optimizer = nnx.Optimizer(
            self.q_network,
            optax.adam(learning_rate),
            wrt=nnx.Param
        )
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Exploration
        self.exploration = EpsilonGreedy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps
        )
```

## Training Loop

```python
def train_dqn(
    num_episodes: int = 300,
    max_steps_per_episode: int = 500,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 10000,
    epsilon_decay_steps: int = 5000,
    hidden_dim: int = 128,
    use_dueling: bool = False,
    seed: int = 42
):
    """Train DQN agent."""
    
    # Create environment and agent
    env = CartPoleEnv(seed=seed)
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay_steps=epsilon_decay_steps,
        use_dueling=use_dueling,
        seed=seed
    )
    
    episode_rewards = []
    
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select and take action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
    
    return agent, episode_rewards
```

## Hyperparameters

### Learning Rate

Controls how quickly the network updates:

| Value | Effect |
|-------|--------|
| 1e-2 | Fast but potentially unstable |
| 1e-3 | Good default for many tasks |
| 1e-4 | Slower, more stable learning |

### Discount Factor (γ)

Determines how much future rewards matter:

| Value | Interpretation |
|-------|----------------|
| 0.99 | Long-term focus (common choice) |
| 0.95 | More myopic |
| 0.9 | Short-term focus |

### Soft Update Rate (τ)

How quickly target network follows main network:

| Value | Effect |
|-------|--------|
| 0.001 | Very smooth updates |
| 0.005 | Common choice |
| 0.01 | Faster target updates |
| 1.0 | Hard update (copy weights) |

### Batch Size

Number of experiences per training step:

| Size | Trade-off |
|------|-----------|
| 32 | Faster updates, noisier gradients |
| 64 | Good balance |
| 128+ | Smoother gradients, slower updates |

### Epsilon Decay

Exploration schedule parameters:

```python
epsilon_start = 1.0    # Start fully random
epsilon_end = 0.01     # Always some exploration
decay_steps = 10000    # Steps to reach epsilon_end
```

## Advanced Techniques

### Double DQN

Reduces overestimation by using online network for action selection:

```python
# Standard DQN: max over target Q-values
next_q_max = jnp.max(self.target_network(next_states), axis=-1)

# Double DQN: select action with online, evaluate with target
best_actions = jnp.argmax(self.q_network(next_states), axis=-1)
next_q_max = self.target_network(next_states)[
    jnp.arange(batch_size), best_actions
]
```

### Prioritized Experience Replay

Sample important transitions more frequently:

```python
class PrioritizedReplayBuffer:
    """Sample based on TD error magnitude."""
    
    def sample(self, batch_size):
        # Higher TD error → higher priority → sampled more often
        priorities = self.priorities ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Importance sampling weights for unbiased updates
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
```

### Noisy Networks

Replace epsilon-greedy with learned exploration:

```python
class NoisyLinear(nnx.Module):
    """Linear layer with learnable noise for exploration."""
    
    def __init__(self, in_features, out_features, *, rngs):
        self.mu_w = nnx.Param(...)
        self.sigma_w = nnx.Param(...)
        self.mu_b = nnx.Param(...)
        self.sigma_b = nnx.Param(...)
    
    def __call__(self, x):
        # Add noise during forward pass
        weight = self.mu_w + self.sigma_w * noise_w
        bias = self.mu_b + self.sigma_b * noise_b
        return x @ weight + bias
```

## Common Pitfalls

### 1. Reward Scaling

❌ **Problem**: Rewards too large or variable  
✅ **Solution**: Clip or normalize rewards

```python
# Clip rewards to [-1, 1]
reward = np.clip(reward, -1, 1)

# Or normalize over running statistics
reward = (reward - running_mean) / (running_std + 1e-8)
```

### 2. Insufficient Exploration

❌ **Problem**: Agent converges to suboptimal policy  
✅ **Solution**: Slower epsilon decay, larger minimum epsilon

```python
# More exploration
epsilon_decay_steps = 50000  # Instead of 10000
epsilon_end = 0.05           # Instead of 0.01
```

### 3. Unstable Learning

❌ **Problem**: Training loss oscillates wildly  
✅ **Solution**: Smaller learning rate, larger replay buffer

```python
learning_rate = 1e-4        # Instead of 1e-3
buffer_size = 500000        # Instead of 100000
tau = 0.001                 # Slower target updates
```

### 4. Forgetting

❌ **Problem**: Agent forgets earlier learning  
✅ **Solution**: Larger buffer, prioritized replay

## Running the Example

Train DQN on CartPole:

```bash
cd examples
python advanced/dqn_reinforcement_learning.py
```

Expected output:
```
DEEP Q-NETWORK (DQN) TRAINING
======================================================================
Architecture: Standard DQN
Episodes: 200
Learning rate: 0.001
Gamma: 0.99
Batch size: 64
======================================================================

Episode   20 | Avg Reward:   23.4 | Best:   45.0 | Epsilon: 0.820 | Buffer:   468
Episode   40 | Avg Reward:   28.7 | Best:   67.0 | Epsilon: 0.640 | Buffer:  1043
Episode   60 | Avg Reward:   45.2 | Best:  112.0 | Epsilon: 0.460 | Buffer:  1947
Episode   80 | Avg Reward:   89.3 | Best:  198.0 | Epsilon: 0.280 | Buffer:  3733
...

TRAINING COMPLETE
======================================================================
Total time: 45.32s
Best episode reward: 500.0
Final average (last 20): 487.3
======================================================================
```

## Extensions

### For More Complex Environments

1. **Convolutional networks** for image observations
2. **Recurrent networks** (DRQN) for partial observability
3. **Distributional RL** (C51, QR-DQN) for better value estimation

### Continuous Action Spaces

DQN is for discrete actions. For continuous:
- **DDPG**: Deterministic actor-critic
- **SAC**: Soft actor-critic (entropy-regularized)
- **TD3**: Twin delayed DDPG

### Multi-Agent RL

- **QMIX**: Coordinated multi-agent learning
- **MAPPO**: Multi-agent PPO

## Complete Example

**Full DQN implementation with all components:**
- [`examples/advanced/dqn_reinforcement_learning.py`](https://github.com/mlnomadpy/flaxdocs/blob/main/examples/advanced/dqn_reinforcement_learning.py)

## References

- **DQN Paper**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
- **Nature DQN**: [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236) (Mnih et al., 2015)
- **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2016)
- **Dueling DQN**: [Dueling Network Architectures](https://arxiv.org/abs/1511.06581) (Wang et al., 2016)
- **Prioritized Replay**: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (Schaul et al., 2016)
- **Rainbow**: [Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) (Hessel et al., 2018)
