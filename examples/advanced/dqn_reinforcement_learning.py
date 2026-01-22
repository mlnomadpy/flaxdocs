"""
Flax NNX: Deep Q-Network (DQN) Reinforcement Learning
======================================================
Implementation of Deep Q-Learning for discrete action spaces.

This example demonstrates:
- Q-Network architecture with Flax NNX
- Experience replay buffer
- Target network with soft updates
- Epsilon-greedy exploration
- Training loop for reinforcement learning
- Integration with gymnasium and gymnax for RL environments
- JAX-native rollouts using jax.lax.scan with gymnax
- Video recording of trained agent simulation

Run: python advanced/dqn_reinforcement_learning.py

Reference:
    Mnih et al. "Playing Atari with Deep Reinforcement Learning"
    DeepMind 2013. https://arxiv.org/abs/1312.5602
    
    Mnih et al. "Human-level control through deep reinforcement learning"
    Nature 2015. https://doi.org/10.1038/nature14236
    
    Lange "gymnax: Classic Gym Environments in JAX"
    GitHub 2022. https://github.com/RobertTLange/gymnax
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, List, Tuple, NamedTuple, Optional, Any
import time
from collections import deque
import random

# Import gymnasium (optional, for video recording)
try:
    import gymnasium as gym
    HAS_GYMNASIUM = True
except ImportError:
    HAS_GYMNASIUM = False
    gym = None

# Import gymnax for JAX-native environments
try:
    import gymnax
    HAS_GYMNAX = True
except ImportError:
    HAS_GYMNAX = False
    gymnax = None

import sys
from pathlib import Path

# Add parent directory to path for shared imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# 1. Q-NETWORK ARCHITECTURE
# ============================================================================

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


class DuelingQNetwork(nnx.Module):
    """
    Dueling DQN architecture that separates value and advantage streams.
    
    Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
    
    Reference:
        Wang et al. "Dueling Network Architectures for Deep Reinforcement Learning"
        ICML 2016. https://arxiv.org/abs/1511.06581
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
        # Shared feature layer
        self.feature = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        
        # Value stream: V(s)
        self.value_fc = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.value_out = nnx.Linear(hidden_dim // 2, 1, rngs=rngs)
        
        # Advantage stream: A(s, a)
        self.advantage_fc = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.advantage_out = nnx.Linear(hidden_dim // 2, action_dim, rngs=rngs)
    
    def __call__(self, state: jax.Array) -> jax.Array:
        """
        Compute Q-values using dueling architecture.
        
        Args:
            state: State observation of shape (batch, state_dim)
            
        Returns:
            Q-values of shape (batch, action_dim)
        """
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


class NoisyLinear(nnx.Module):
    """
    Noisy Linear layer for exploration via parameter noise.
    
    Reference:
        Fortunato et al. "Noisy Networks for Exploration"
        ICLR 2018. https://arxiv.org/abs/1706.10295
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        sigma_init: float = 0.5,
        *,
        rngs: nnx.Rngs
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        mu_range = 1 / np.sqrt(in_features)
        self.weight_mu = nnx.Param(
            jax.random.uniform(rngs.params(), (out_features, in_features), minval=-mu_range, maxval=mu_range)
        )
        self.weight_sigma = nnx.Param(
            jnp.full((out_features, in_features), sigma_init / np.sqrt(in_features))
        )
        self.bias_mu = nnx.Param(
            jax.random.uniform(rngs.params(), (out_features,), minval=-mu_range, maxval=mu_range)
        )
        self.bias_sigma = nnx.Param(
            jnp.full((out_features,), sigma_init / np.sqrt(out_features))
        )
        
        self.rngs = rngs
    
    def __call__(self, x: jax.Array) -> jax.Array:
        # Generate noise
        weight_epsilon = jax.random.normal(self.rngs.noise(), (self.out_features, self.in_features))
        bias_epsilon = jax.random.normal(self.rngs.noise(), (self.out_features,))
        
        # Noisy weights (using direct array access to avoid deprecation warnings)
        weight = self.weight_mu[...] + self.weight_sigma[...] * weight_epsilon
        bias = self.bias_mu[...] + self.bias_sigma[...] * bias_epsilon
        
        return x @ weight.T + bias


class NoisyQNetwork(nnx.Module):
    """
    NoisyNet DQN that uses noisy layers for exploration.
    
    Replaces epsilon-greedy with learned exploration through parameter noise.
    
    Reference:
        Fortunato et al. "Noisy Networks for Exploration"
        ICLR 2018. https://arxiv.org/abs/1706.10295
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        rngs: nnx.Rngs
    ):
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = NoisyLinear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = NoisyLinear(hidden_dim, action_dim, rngs=rngs)
    
    def __call__(self, state: jax.Array) -> jax.Array:
        x = self.fc1(state)
        x = nnx.relu(x)
        x = self.fc2(x)
        x = nnx.relu(x)
        q_values = self.fc3(x)
        return q_values


class DeepQNetwork(nnx.Module):
    """
    Deep Q-Network with more layers for complex environments.
    
    Has 4 hidden layers instead of 2 for potentially better representation.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        *,
        rngs: nnx.Rngs
    ):
        self.fc1 = nnx.Linear(state_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.fc3 = nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs)
        self.fc4 = nnx.Linear(hidden_dim // 2, hidden_dim // 2, rngs=rngs)
        self.fc5 = nnx.Linear(hidden_dim // 2, action_dim, rngs=rngs)
    
    def __call__(self, state: jax.Array) -> jax.Array:
        x = nnx.relu(self.fc1(state))
        x = nnx.relu(self.fc2(x))
        x = nnx.relu(self.fc3(x))
        x = nnx.relu(self.fc4(x))
        return self.fc5(x)


# ============================================================================
# 2. EXPERIENCE REPLAY BUFFER
# ============================================================================

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
        """
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Dict[str, jax.Array]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with batched states, actions, rewards, next_states, dones
        """
        transitions = random.sample(self.buffer, batch_size)
        
        # Convert to batched arrays
        states = jnp.array([t.state for t in transitions])
        actions = jnp.array([t.action for t in transitions])
        rewards = jnp.array([t.reward for t in transitions])
        next_states = jnp.array([t.next_state for t in transitions])
        dones = jnp.array([t.done for t in transitions], dtype=jnp.float32)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    
    Samples important transitions more frequently based on TD error.
    
    Reference:
        Schaul et al. "Prioritized Experience Replay"
        ICLR 2016. https://arxiv.org/abs/1511.05952
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta: float = 0.4, beta_increment: float = 0.001):
        """
        Args:
            capacity: Maximum buffer size
            alpha: How much prioritization to use (0 = uniform, 1 = full)
            beta: Importance sampling correction (0 = no correction, 1 = full)
            beta_increment: How much to increase beta per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Add transition with maximum priority."""
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[Dict[str, jax.Array], np.ndarray, np.ndarray]:
        """
        Sample transitions based on priorities.
        
        Returns:
            batch: Dictionary of batched transitions
            indices: Indices of sampled transitions
            weights: Importance sampling weights
        """
        # Increase beta toward 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Compute sampling probabilities
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Get transitions
        transitions = [self.buffer[i] for i in indices]
        
        batch = {
            'states': jnp.array([t.state for t in transitions]),
            'actions': jnp.array([t.action for t in transitions]),
            'rewards': jnp.array([t.reward for t in transitions]),
            'next_states': jnp.array([t.next_state for t in transitions]),
            'dones': jnp.array([t.done for t in transitions], dtype=jnp.float32)
        }
        
        return batch, indices, jnp.array(weights)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self) -> int:
        return len(self.buffer)


# ============================================================================
# 3. EPSILON-GREEDY EXPLORATION
# ============================================================================

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
        """
        Args:
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps to decay epsilon from start to end
        """
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.step_count = 0
    
    @property
    def epsilon(self) -> float:
        """Current epsilon value with linear decay."""
        progress = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
    
    def select_action(
        self,
        q_values: jax.Array,
        num_actions: int
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            q_values: Q-values for current state, shape (action_dim,)
            num_actions: Number of possible actions
            
        Returns:
            Selected action index
        """
        self.step_count += 1
        
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, num_actions - 1)
        else:
            # Exploit: best action
            return int(jnp.argmax(q_values))


# ============================================================================
# 4. DQN AGENT
# ============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent.
    
    Combines all DQN components:
    - Q-Network for action value estimation
    - Target network for stable learning
    - Experience replay for decorrelated training
    - Epsilon-greedy for exploration
    
    Supported architectures: 'standard', 'dueling', 'noisy', 'deep'
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
        architecture: str = None,
        seed: int = 0
    ):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer size
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            tau: Soft update coefficient for target network
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay_steps: Steps for epsilon decay
            use_dueling: Whether to use dueling architecture (deprecated, use architecture='dueling')
            architecture: Network architecture: 'standard', 'dueling', 'noisy', 'deep'
            seed: Random seed
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Resolve architecture (architecture param takes precedence over use_dueling)
        if architecture is not None:
            self.architecture = architecture
        elif use_dueling:
            self.architecture = 'dueling'
        else:
            self.architecture = 'standard'
        
        # Initialize networks
        rngs = nnx.Rngs(seed)
        
        # Select network architecture
        network_classes = {
            'standard': QNetwork,
            'dueling': DuelingQNetwork,
            'noisy': NoisyQNetwork,
            'deep': DeepQNetwork
        }
        NetworkClass = network_classes.get(self.architecture, QNetwork)
        
        self.q_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rngs=rngs
        )
        
        # Create target network (copy of q_network)
        self.target_network = NetworkClass(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            rngs=nnx.Rngs(seed + 1)  # Different seed, will copy params
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
        
        # Exploration strategy
        self.exploration = EpsilonGreedy(
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps
        )
        
        # Training statistics
        self.total_steps = 0
        self.training_losses = []
    
    def _hard_update_target(self):
        """Copy Q-network parameters to target network."""
        q_params = nnx.state(self.q_network, nnx.Param)
        target_params = nnx.state(self.target_network, nnx.Param)
        
        # Copy all parameters
        new_target_params = jax.tree.map(lambda q, t: q, q_params, target_params)
        nnx.update(self.target_network, new_target_params)
    
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
    
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        Select action for given state.
        
        Args:
            state: Current state observation
            eval_mode: If True, always take greedy action
            
        Returns:
            Action index
        """
        # Get Q-values
        state_tensor = jnp.array(state).reshape(1, -1)
        q_values = self.q_network(state_tensor)[0]
        
        if eval_mode:
            return int(jnp.argmax(q_values))
        else:
            return self.exploration.select_action(q_values, self.action_dim)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def train_step(self) -> float:
        """
        Perform one training step.
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Compute loss and update
        loss = self._update(batch)
        
        # Soft update target network
        self._soft_update_target()
        
        self.training_losses.append(float(loss))
        return float(loss)
    
    def _update(self, batch: Dict[str, jax.Array]) -> jax.Array:
        """
        Compute TD loss and update Q-network.
        
        Uses the Bellman equation:
        Q(s, a) ← r + γ * max_a' Q_target(s', a')
        """
        gamma = self.gamma
        target_network = self.target_network
        
        def loss_fn(model):
            # Current Q-values for taken actions
            q_values = model(batch['states'])
            q_values_selected = q_values[
                jnp.arange(len(batch['actions'])),
                batch['actions']
            ]
            
            # Target Q-values (no gradient through target network)
            next_q_values = target_network(batch['next_states'])
            next_q_max = jnp.max(next_q_values, axis=-1)
            
            # TD target: r + γ * max Q(s', a') * (1 - done)
            targets = batch['rewards'] + gamma * next_q_max * (1 - batch['dones'])
            
            # Mean squared TD error
            td_errors = q_values_selected - targets
            loss = jnp.mean(td_errors ** 2)
            
            return loss
        
        # Compute gradients and update
        loss, grads = nnx.value_and_grad(loss_fn)(self.q_network)
        self.optimizer.update(self.q_network, grads)
        
        return loss


# ============================================================================
# 5. GYMNASIUM ENVIRONMENT WRAPPER
# ============================================================================

class CartPoleEnv:
    """
    Wrapper around gymnasium's CartPole-v1 environment.
    
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left), 1 (push right)
    
    This wrapper provides a consistent interface for the DQN agent.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        if not HAS_GYMNASIUM:
            raise ImportError(
                "gymnasium is not installed. Install with: "
                "pip install gymnasium[classic-control]"
            )
        self.env = gym.make('CartPole-v1')
        self._seed = seed
        if seed is not None:
            self.env.reset(seed=seed)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        state, _ = self.env.reset(seed=self._seed)
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: 0 (left) or 1 (right)
            
        Returns:
            next_state, reward, done, info
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return state.astype(np.float32), float(reward), done, info
    
    def close(self):
        """Clean up the environment."""
        self.env.close()
    
    @property
    def state_dim(self) -> int:
        return 4
    
    @property
    def action_dim(self) -> int:
        return 2


class GymnaxCartPoleEnv:
    """
    Wrapper around gymnax's CartPole-v1 environment for JAX-native RL.
    
    gymnax provides fully JAX-compatible environments that support:
    - JIT compilation for fast training
    - vmap for parallel environment execution
    - Full compatibility with JAX transformations
    
    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
    Actions: 0 (push left), 1 (push right)
    
    Reference:
        Lange "gymnax: Classic Gym Environments in JAX"
        https://github.com/RobertTLange/gymnax
    
    Example usage:
        >>> env = GymnaxCartPoleEnv(seed=42)
        >>> state, obs = env.reset()
        >>> next_obs, next_state, reward, done, info = env.step(state, action)
    """
    
    def __init__(self, seed: int = 0):
        """
        Initialize gymnax CartPole-v1 environment.
        
        Args:
            seed: Random seed for reproducibility
        """
        if not HAS_GYMNAX:
            raise ImportError(
                "gymnax is not installed. Install with: "
                "pip install gymnax"
            )
        
        self.env, self.env_params = gymnax.make("CartPole-v1")
        self.key = jax.random.key(seed)
        self._seed = seed
    
    def reset(self) -> Tuple[Any, jax.Array]:
        """
        Reset environment to initial state.
        
        Returns:
            state: Internal environment state (for gymnax step function)
            obs: Observable state as JAX array (4,)
        """
        self.key, key_reset = jax.random.split(self.key)
        obs, state = self.env.reset(key_reset, self.env_params)
        return state, obs
    
    def step(
        self,
        state: Any,
        action: jax.Array
    ) -> Tuple[jax.Array, Any, jax.Array, jax.Array, dict]:
        """
        Take a step in the environment (JAX-compatible).
        
        Args:
            state: Current environment state
            action: Action to take (0 or 1)
            
        Returns:
            next_obs: Next observation (4,)
            next_state: Next internal state
            reward: Reward received
            done: Whether episode is done
            info: Additional info dict
        """
        self.key, key_step = jax.random.split(self.key)
        next_obs, next_state, reward, done, info = self.env.step(
            key_step, state, action, self.env_params
        )
        return next_obs, next_state, reward, done, info
    
    def step_numpy(
        self,
        state: Any,
        action: int
    ) -> Tuple[np.ndarray, Any, float, bool, dict]:
        """
        Take a step in the environment (NumPy-compatible for standard training).
        
        Args:
            state: Current environment state
            action: Action to take (0 or 1)
            
        Returns:
            next_obs: Next observation as numpy array
            next_state: Next internal state
            reward: Reward as float
            done: Whether episode is done
            info: Additional info dict
        """
        next_obs, next_state, reward, done, info = self.step(state, jnp.array(action))
        return (
            np.asarray(next_obs, dtype=np.float32),
            next_state,
            float(reward),
            bool(done),
            info
        )
    
    def reset_numpy(self) -> Tuple[Any, np.ndarray]:
        """
        Reset environment (NumPy-compatible for standard training).
        
        Returns:
            state: Internal environment state
            obs: Observable state as numpy array
        """
        state, obs = self.reset()
        return state, np.asarray(obs, dtype=np.float32)
    
    def close(self):
        """No cleanup needed for gymnax environments."""
        pass
    
    @property
    def state_dim(self) -> int:
        return 4
    
    @property
    def action_dim(self) -> int:
        return 2


def gymnax_rollout(
    key: jax.Array,
    policy_fn,
    policy_params: Any,
    steps_in_episode: int,
    env_name: str = "CartPole-v1"
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Perform a JAX-native episode rollout using jax.lax.scan.
    
    This function demonstrates the power of gymnax for fully JIT-compiled
    rollouts, enabling fast training and evaluation.
    
    Args:
        key: JAX random key
        policy_fn: Policy function: (params, obs, key) -> action
        policy_params: Parameters for policy function
        steps_in_episode: Maximum number of steps
        env_name: Name of gymnax environment
        
    Returns:
        obs: Observations of shape (steps, obs_dim)
        actions: Actions of shape (steps,)
        rewards: Rewards of shape (steps,)
        next_obs: Next observations of shape (steps, obs_dim)
        dones: Done flags of shape (steps,)
        
    Example:
        >>> from flax import linen as nn
        >>> class Policy(nn.Module):
        ...     @nn.compact
        ...     def __call__(self, x, key):
        ...         return nn.Dense(2)(x)
        >>> model = Policy()
        >>> params = model.init(jax.random.key(0), jnp.zeros(4), None)
        >>> obs, actions, rewards, next_obs, dones = gymnax_rollout(
        ...     jax.random.key(42), model.apply, params, 200
        ... )
    """
    if not HAS_GYMNAX:
        raise ImportError(
            "gymnax is not installed. Install with: "
            "pip install gymnax"
        )
    
    env, env_params = gymnax.make(env_name)
    
    # Reset the environment
    key, key_reset, key_episode = jax.random.split(key, 3)
    obs, state = env.reset(key_reset, env_params)
    
    def policy_step(state_input, _):
        """Step transition in JAX env using jax.lax.scan."""
        obs, state, policy_params, key = state_input
        key, key_step, key_net = jax.random.split(key, 3)
        
        # Get action from policy
        action = policy_fn(policy_params, obs, key_net)
        # For discrete actions, take argmax if output is logits (multi-dimensional output)
        action_shape = getattr(action, 'shape', ())
        if action_shape and action_shape[-1] > 1:
            action = jnp.argmax(action, axis=-1)
        
        # Step environment
        next_obs, next_state, reward, done, _ = env.step(
            key_step, state, action, env_params
        )
        
        carry = [next_obs, next_state, policy_params, key]
        return carry, [obs, action, reward, next_obs, done]
    
    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step,
        [obs, state, policy_params, key_episode],
        None,  # No input needed, just counting
        steps_in_episode
    )
    
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


def gymnax_vmap_rollout(
    keys: jax.Array,
    policy_fn,
    policy_params: Any,
    steps_in_episode: int,
    env_name: str = "CartPole-v1"
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Perform parallel episode rollouts using vmap.
    
    This enables training on multiple environments simultaneously,
    significantly speeding up data collection.
    
    Args:
        keys: JAX random keys of shape (num_envs,)
        policy_fn: Policy function: (params, obs, key) -> action
        policy_params: Parameters for policy function
        steps_in_episode: Maximum number of steps
        env_name: Name of gymnax environment
        
    Returns:
        obs: Observations of shape (num_envs, steps, obs_dim)
        actions: Actions of shape (num_envs, steps)
        rewards: Rewards of shape (num_envs, steps)
        next_obs: Next observations of shape (num_envs, steps, obs_dim)
        dones: Done flags of shape (num_envs, steps)
        
    Example:
        >>> keys = jax.random.split(jax.random.key(0), 8)  # 8 parallel envs
        >>> obs, actions, rewards, next_obs, dones = gymnax_vmap_rollout(
        ...     keys, model.apply, params, 200
        ... )
        >>> print(obs.shape)  # (8, 200, 4)
    """
    vmap_rollout = jax.vmap(
        gymnax_rollout,
        in_axes=(0, None, None, None, None)
    )
    return vmap_rollout(keys, policy_fn, policy_params, steps_in_episode, env_name)


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

def train_dqn_gymnax(
    num_episodes: int = 300,
    max_steps_per_episode: int = 500,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_size: int = 10000,
    epsilon_decay_steps: int = 5000,
    hidden_dim: int = 128,
    use_dueling: bool = False,
    architecture: str = None,
    eval_frequency: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DQNAgent, List[float], Dict]:
    """
    Train DQN agent on CartPole environment using gymnax (JAX-native).
    
    This function uses gymnax for JAX-compatible environment simulation,
    enabling faster training through JIT compilation.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        learning_rate: Learning rate
        gamma: Discount factor
        batch_size: Training batch size
        buffer_size: Replay buffer size
        epsilon_decay_steps: Steps for epsilon decay
        hidden_dim: Network hidden dimension
        use_dueling: Whether to use dueling architecture (deprecated, use architecture)
        architecture: Network architecture: 'standard', 'dueling', 'noisy', 'deep'
        eval_frequency: Episodes between evaluations
        seed: Random seed
        verbose: Whether to print training progress
        
    Returns:
        Trained agent, episode rewards, and training metrics dict
        
    Example:
        >>> agent, rewards, metrics = train_dqn_gymnax(
        ...     num_episodes=100,
        ...     architecture='standard',
        ...     seed=42
        ... )
        >>> print(f"Final avg reward: {metrics['final_avg_reward']:.1f}")
    """
    if not HAS_GYMNAX:
        raise ImportError(
            "gymnax is not installed. Install with: "
            "pip install gymnax"
        )
    
    # Create gymnax environment
    env = GymnaxCartPoleEnv(seed=seed)
    
    arch = architecture or ('dueling' if use_dueling else 'standard')
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay_steps=epsilon_decay_steps,
        architecture=arch,
        seed=seed
    )
    
    episode_rewards = []
    best_reward = 0
    training_times = []
    
    arch_names = {
        'standard': 'Standard DQN',
        'dueling': 'Dueling DQN',
        'noisy': 'NoisyNet DQN',
        'deep': 'Deep DQN (4-layer)'
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("DEEP Q-NETWORK (DQN) TRAINING WITH GYMNAX")
        print(f"{'='*70}")
        print(f"Architecture: {arch_names.get(arch, arch)}")
        print(f"Environment: gymnax CartPole-v1 (JAX-native)")
        print(f"Episodes: {num_episodes}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gamma: {gamma}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Reset environment (gymnax returns state and obs)
        env_state, obs = env.reset_numpy()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # Select and take action
            action = agent.select_action(obs)
            next_obs, env_state, reward, done, _ = env.step_numpy(env_state, action)
            
            # Store transition
            agent.store_transition(obs, action, reward, next_obs, done)
            
            # Train
            agent.train_step()
            
            episode_reward += reward
            obs = next_obs
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        training_times.append(time.time() - start_time)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if verbose and episode % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Best: {best_reward:6.1f} | "
                  f"Epsilon: {agent.exploration.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE (gymnax)")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best episode reward: {best_reward:.1f}")
        print(f"Final average (last 20): {np.mean(episode_rewards[-20:]):.1f}")
        print(f"{'='*70}\n")
    
    # Compute model metrics
    params = nnx.state(agent.q_network, nnx.Param)
    num_params = sum(p.size for p in jax.tree.leaves(params))
    
    metrics = {
        'architecture': arch,
        'environment': 'gymnax',
        'total_time': total_time,
        'best_reward': best_reward,
        'final_avg_reward': float(np.mean(episode_rewards[-20:])),
        'num_params': num_params,
        'training_times': training_times,
        'episode_per_second': num_episodes / total_time
    }
    
    env.close()
    return agent, episode_rewards, metrics


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
    architecture: str = None,
    eval_frequency: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[DQNAgent, List[float], Dict]:
    """
    Train DQN agent on CartPole environment.
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        learning_rate: Learning rate
        gamma: Discount factor
        batch_size: Training batch size
        buffer_size: Replay buffer size
        epsilon_decay_steps: Steps for epsilon decay
        hidden_dim: Network hidden dimension
        use_dueling: Whether to use dueling architecture (deprecated, use architecture)
        architecture: Network architecture: 'standard', 'dueling', 'noisy', 'deep'
        eval_frequency: Episodes between evaluations
        seed: Random seed
        verbose: Whether to print training progress
        
    Returns:
        Trained agent, episode rewards, and training metrics dict
    """
    # Create environment and agent
    env = CartPoleEnv(seed=seed)
    
    arch = architecture or ('dueling' if use_dueling else 'standard')
    
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        epsilon_decay_steps=epsilon_decay_steps,
        architecture=arch,
        seed=seed
    )
    
    episode_rewards = []
    best_reward = 0
    training_times = []
    
    arch_names = {
        'standard': 'Standard DQN',
        'dueling': 'Dueling DQN',
        'noisy': 'NoisyNet DQN',
        'deep': 'Deep DQN (4-layer)'
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("DEEP Q-NETWORK (DQN) TRAINING")
        print(f"{'='*70}")
        print(f"Architecture: {arch_names.get(arch, arch)}")
        print(f"Episodes: {num_episodes}")
        print(f"Learning rate: {learning_rate}")
        print(f"Gamma: {gamma}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*70}\n")
    
    start_time = time.time()
    
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
        training_times.append(time.time() - start_time)
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if verbose and episode % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Best: {best_reward:6.1f} | "
                  f"Epsilon: {agent.exploration.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    if verbose:
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Best episode reward: {best_reward:.1f}")
        print(f"Final average (last 20): {np.mean(episode_rewards[-20:]):.1f}")
        print(f"{'='*70}\n")
    
    # Compute model metrics
    params = nnx.state(agent.q_network, nnx.Param)
    num_params = sum(p.size for p in jax.tree.leaves(params))
    
    metrics = {
        'architecture': arch,
        'total_time': total_time,
        'best_reward': best_reward,
        'final_avg_reward': float(np.mean(episode_rewards[-20:])),
        'num_params': num_params,
        'training_times': training_times,
        'episode_per_second': num_episodes / total_time
    }
    
    env.close()
    return agent, episode_rewards, metrics


def evaluate_agent(agent: DQNAgent, num_episodes: int = 10, seed: int = 0) -> float:
    """
    Evaluate trained agent using gymnasium.
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of evaluation episodes
        seed: Random seed
        
    Returns:
        Average reward across episodes
    """
    env = CartPoleEnv(seed=seed)
    rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(state, eval_mode=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
    
    env.close()
    return np.mean(rewards)


def evaluate_agent_gymnax(
    agent: DQNAgent,
    num_episodes: int = 10,
    seed: int = 0,
    max_steps: int = 500
) -> float:
    """
    Evaluate trained agent using gymnax (JAX-native).
    
    Args:
        agent: Trained DQN agent
        num_episodes: Number of evaluation episodes
        seed: Random seed
        max_steps: Maximum steps per episode
        
    Returns:
        Average reward across episodes
        
    Example:
        >>> agent, _, _ = train_dqn_gymnax(num_episodes=100)
        >>> avg_reward = evaluate_agent_gymnax(agent, num_episodes=10)
        >>> print(f"Average reward: {avg_reward:.1f}")
    """
    if not HAS_GYMNAX:
        raise ImportError(
            "gymnax is not installed. Install with: "
            "pip install gymnax"
        )
    
    env = GymnaxCartPoleEnv(seed=seed)
    rewards = []
    
    for episode in range(num_episodes):
        env_state, obs = env.reset_numpy()
        episode_reward = 0
        
        for _ in range(max_steps):
            action = agent.select_action(obs, eval_mode=True)
            obs, env_state, reward, done, _ = env.step_numpy(env_state, action)
            episode_reward += reward
            
            if done:
                break
        
        rewards.append(episode_reward)
    
    env.close()
    return np.mean(rewards)


# ============================================================================
# 7. VIDEO RECORDING AND VISUALIZATION
# ============================================================================

def record_video(
    agent: DQNAgent,
    video_folder: str = "videos",
    video_name: str = "dqn_cartpole",
    num_episodes: int = 3,
    max_steps: int = 500,
    seed: int = 0
) -> Optional[str]:
    """
    Record video of trained agent playing CartPole.
    
    Uses gymnasium's RecordVideo wrapper to capture the simulation
    and save it as an MP4 video file.
    
    Args:
        agent: Trained DQN agent
        video_folder: Directory to save videos
        video_name: Name prefix for the video file
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        seed: Random seed for reproducibility
        
    Returns:
        Path to the saved video file, or None if video creation failed
        
    Example:
        >>> agent, _ = train_dqn(num_episodes=100)
        >>> video_path = record_video(agent, video_folder="./my_videos")
        >>> if video_path:
        ...     print(f"Video saved to: {video_path}")
    """
    from gymnasium.wrappers import RecordVideo
    import os
    
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Create environment with rgb_array render mode for video recording
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    
    # Wrap with RecordVideo - records all episodes
    env = RecordVideo(
        env, 
        video_folder=video_folder,
        name_prefix=video_name,
        episode_trigger=lambda x: True  # Record all episodes
    )
    
    print(f"\n🎬 Recording {num_episodes} episodes...")
    
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state = state.astype(np.float32)
        episode_reward = 0
        
        for step in range(max_steps):
            # Get action from trained agent (greedy/eval mode)
            action = agent.select_action(state, eval_mode=True)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state.astype(np.float32)
            
            if done:
                break
        
        total_rewards.append(episode_reward)
        print(f"  Episode {episode + 1}/{num_episodes}: Reward = {episode_reward:.0f}")
    
    env.close()
    
    # Find the recorded video file(s)
    video_files = sorted([f for f in os.listdir(video_folder) if f.startswith(video_name) and f.endswith('.mp4')])
    
    if video_files:
        video_path = os.path.join(video_folder, video_files[-1])
        print(f"\n✅ Video saved to: {video_path}")
    else:
        video_path = None
        print(f"\n⚠️ No video files found in: {video_folder}")
    
    print(f"   Average reward: {np.mean(total_rewards):.1f}")
    
    return video_path


def plot_training_progress(
    episode_rewards: List[float],
    window_size: int = 20,
    save_path: str = None,
    title: str = "DQN Training Progress"
) -> None:
    """
    Plot training progress with episode rewards and moving average.
    
    Args:
        episode_rewards: List of rewards per episode
        window_size: Window size for moving average
        save_path: Optional path to save the plot image
        title: Title for the plot
        
    Example:
        >>> agent, rewards = train_dqn(num_episodes=200)
        >>> plot_training_progress(rewards, save_path="training_plot.png")
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        print("Install with: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    episodes = range(1, len(episode_rewards) + 1)
    
    # Plot raw rewards
    ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Episode Reward')
    
    # Calculate and plot moving average
    if len(episode_rewards) >= window_size:
        moving_avg = []
        for i in range(len(episode_rewards)):
            if i < window_size:
                moving_avg.append(np.mean(episode_rewards[:i+1]))
            else:
                moving_avg.append(np.mean(episode_rewards[i-window_size+1:i+1]))
        ax.plot(episodes, moving_avg, color='red', linewidth=2, 
                label=f'Moving Average ({window_size} episodes)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limit based on CartPole max reward
    ax.set_ylim(0, 550)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"📊 Plot saved to: {save_path}")
    
    plt.close(fig)


def compare_architectures(
    architectures: List[str] = None,
    num_episodes: int = 150,
    num_runs: int = 1,
    hidden_dim: int = 128,
    seed: int = 42,
    save_dir: str = "comparison_plots"
) -> Dict:
    """
    Train and compare multiple DQN architectures.
    
    Generates comparison plots for:
    - Training rewards over time
    - Performance (final average reward)
    - Training speed (episodes per second)
    - Model size (number of parameters)
    - Weight statistics (mean, std) over training
    
    Args:
        architectures: List of architectures to compare ('standard', 'dueling', 'noisy', 'deep')
        num_episodes: Number of training episodes per model
        num_runs: Number of runs for averaging (use >1 for statistical significance)
        hidden_dim: Hidden layer dimension
        seed: Base random seed
        save_dir: Directory to save comparison plots
        
    Returns:
        Dictionary with comparison results and metrics
        
    Example:
        >>> results = compare_architectures(['standard', 'dueling', 'deep'], num_episodes=100)
        >>> print(results['summary'])
    """
    import os
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping comparison plots.")
        print("Install with: pip install matplotlib")
        return {}
    
    os.makedirs(save_dir, exist_ok=True)
    
    if architectures is None:
        architectures = ['standard', 'dueling', 'noisy', 'deep']
    
    arch_names = {
        'standard': 'Standard DQN',
        'dueling': 'Dueling DQN',
        'noisy': 'NoisyNet DQN',
        'deep': 'Deep DQN (4-layer)'
    }
    
    colors = {
        'standard': '#1f77b4',
        'dueling': '#ff7f0e',
        'noisy': '#2ca02c',
        'deep': '#d62728'
    }
    
    results = {
        'architectures': architectures,
        'rewards': {},
        'metrics': {},
        'weight_stats': {}
    }
    
    print("\n" + "=" * 70)
    print("DQN ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(f"Architectures: {', '.join([arch_names.get(a, a) for a in architectures])}")
    print(f"Episodes per model: {num_episodes}")
    print(f"Runs per architecture: {num_runs}")
    print("=" * 70 + "\n")
    
    # Train each architecture
    for arch in architectures:
        print(f"\n🔄 Training {arch_names.get(arch, arch)}...")
        
        all_rewards = []
        all_metrics = []
        
        for run in range(num_runs):
            run_seed = seed + run * 100
            agent, rewards, metrics = train_dqn(
                num_episodes=num_episodes,
                architecture=arch,
                hidden_dim=hidden_dim,
                seed=run_seed,
                verbose=(num_runs == 1)
            )
            all_rewards.append(rewards)
            all_metrics.append(metrics)
            
            # Track weight statistics
            params = nnx.state(agent.q_network, nnx.Param)
            param_values = [p.flatten() for p in jax.tree.leaves(params)]
            all_params = jnp.concatenate(param_values)
            
            if arch not in results['weight_stats']:
                results['weight_stats'][arch] = {
                    'mean': [],
                    'std': [],
                    'min': [],
                    'max': []
                }
            results['weight_stats'][arch]['mean'].append(float(jnp.mean(all_params)))
            results['weight_stats'][arch]['std'].append(float(jnp.std(all_params)))
            results['weight_stats'][arch]['min'].append(float(jnp.min(all_params)))
            results['weight_stats'][arch]['max'].append(float(jnp.max(all_params)))
        
        # Average rewards across runs (with empty list protection)
        if all_rewards and all(len(r) > 0 for r in all_rewards):
            max_len = max(len(r) for r in all_rewards)
            padded_rewards = [r + [r[-1]] * (max_len - len(r)) for r in all_rewards]
            avg_rewards = np.mean(padded_rewards, axis=0)
        else:
            avg_rewards = np.array([0.0])  # Default if no rewards
        
        results['rewards'][arch] = avg_rewards.tolist()
        results['metrics'][arch] = {
            'num_params': all_metrics[0]['num_params'],
            'avg_time': np.mean([m['total_time'] for m in all_metrics]),
            'avg_final_reward': np.mean([m['final_avg_reward'] for m in all_metrics]),
            'avg_best_reward': np.mean([m['best_reward'] for m in all_metrics]),
            'episodes_per_second': np.mean([m['episode_per_second'] for m in all_metrics])
        }
    
    # Generate comparison plots
    _plot_reward_comparison(results, arch_names, colors, save_dir)
    _plot_performance_comparison(results, arch_names, colors, save_dir)
    _plot_speed_comparison(results, arch_names, colors, save_dir)
    _plot_size_comparison(results, arch_names, colors, save_dir)
    _plot_weight_dynamics(results, arch_names, colors, save_dir)
    
    # Print summary
    _print_comparison_summary(results, arch_names)
    
    results['summary'] = _generate_summary_dict(results, arch_names)
    
    return results


def _plot_reward_comparison(results: Dict, arch_names: Dict, colors: Dict, save_dir: str):
    """Plot training rewards comparison across architectures."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for arch, rewards in results['rewards'].items():
        episodes = range(1, len(rewards) + 1)
        
        # Calculate moving average
        window = 20
        moving_avg = []
        for i in range(len(rewards)):
            if i < window:
                moving_avg.append(np.mean(rewards[:i+1]))
            else:
                moving_avg.append(np.mean(rewards[i-window+1:i+1]))
        
        ax.plot(episodes, moving_avg, label=arch_names.get(arch, arch), 
                color=colors.get(arch, 'gray'), linewidth=2)
        ax.fill_between(episodes, 
                        np.array(moving_avg) - 20,
                        np.array(moving_avg) + 20,
                        color=colors.get(arch, 'gray'), alpha=0.1)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward (20-episode moving average)', fontsize=12)
    ax.set_title('Training Reward Comparison Across DQN Architectures', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 550)
    
    plt.tight_layout()
    save_path = f"{save_dir}/reward_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Saved: {save_path}")
    plt.close(fig)


def _plot_performance_comparison(results: Dict, arch_names: Dict, colors: Dict, save_dir: str):
    """Plot final performance bar chart."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    archs = list(results['metrics'].keys())
    final_rewards = [results['metrics'][a]['avg_final_reward'] for a in archs]
    best_rewards = [results['metrics'][a]['avg_best_reward'] for a in archs]
    
    x = np.arange(len(archs))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_rewards, width, 
                   label='Final Avg Reward (last 20 ep.)',
                   color=[colors.get(a, 'gray') for a in archs], alpha=0.8)
    bars2 = ax.bar(x + width/2, best_rewards, width,
                   label='Best Episode Reward',
                   color=[colors.get(a, 'gray') for a in archs], alpha=0.5)
    
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('Performance Comparison: Final and Best Rewards', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([arch_names.get(a, a) for a in archs], rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = f"{save_dir}/performance_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Saved: {save_path}")
    plt.close(fig)


def _plot_speed_comparison(results: Dict, arch_names: Dict, colors: Dict, save_dir: str):
    """Plot training speed comparison."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    archs = list(results['metrics'].keys())
    times = [results['metrics'][a]['avg_time'] for a in archs]
    eps_per_sec = [results['metrics'][a]['episodes_per_second'] for a in archs]
    
    # Total training time
    bars1 = ax1.bar(range(len(archs)), times, 
                    color=[colors.get(a, 'gray') for a in archs])
    ax1.set_xlabel('Architecture', fontsize=12)
    ax1.set_ylabel('Training Time (seconds)', fontsize=12)
    ax1.set_title('Total Training Time', fontsize=14)
    ax1.set_xticks(range(len(archs)))
    ax1.set_xticklabels([arch_names.get(a, a) for a in archs], rotation=15)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}s',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    # Episodes per second
    bars2 = ax2.bar(range(len(archs)), eps_per_sec,
                    color=[colors.get(a, 'gray') for a in archs])
    ax2.set_xlabel('Architecture', fontsize=12)
    ax2.set_ylabel('Episodes per Second', fontsize=12)
    ax2.set_title('Training Speed', fontsize=14)
    ax2.set_xticks(range(len(archs)))
    ax2.set_xticklabels([arch_names.get(a, a) for a in archs], rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = f"{save_dir}/speed_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Saved: {save_path}")
    plt.close(fig)


def _plot_size_comparison(results: Dict, arch_names: Dict, colors: Dict, save_dir: str):
    """Plot model size (parameter count) comparison."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    archs = list(results['metrics'].keys())
    params = [results['metrics'][a]['num_params'] for a in archs]
    
    bars = ax.bar(range(len(archs)), params,
                  color=[colors.get(a, 'gray') for a in archs])
    
    ax.set_xlabel('Architecture', fontsize=12)
    ax.set_ylabel('Number of Parameters', fontsize=12)
    ax.set_title('Model Size Comparison', fontsize=14)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([arch_names.get(a, a) for a in archs], rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = f"{save_dir}/size_comparison.png"
    plt.savefig(save_path, dpi=150)
    print(f"📊 Saved: {save_path}")
    plt.close(fig)


def _plot_weight_dynamics(results: Dict, arch_names: Dict, colors: Dict, save_dir: str):
    """Plot weight statistics comparison."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    archs = list(results['weight_stats'].keys())
    metrics_names = ['mean', 'std', 'min', 'max']
    titles = ['Weight Mean', 'Weight Std Dev', 'Weight Min', 'Weight Max']
    
    for idx, (metric, title) in enumerate(zip(metrics_names, titles)):
        ax = axes[idx // 2, idx % 2]
        
        values = [np.mean(results['weight_stats'][a][metric]) for a in archs]
        
        bars = ax.bar(range(len(archs)), values,
                      color=[colors.get(a, 'gray') for a in archs])
        
        ax.set_xlabel('Architecture', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'Final {title}', fontsize=12)
        ax.set_xticks(range(len(archs)))
        ax.set_xticklabels([arch_names.get(a, a) for a in archs], rotation=15, fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Weight Dynamics Comparison After Training', fontsize=14, y=1.02)
    plt.tight_layout()
    save_path = f"{save_dir}/weight_dynamics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {save_path}")
    plt.close(fig)


def _print_comparison_summary(results: Dict, arch_names: Dict):
    """Print a summary table of comparison results."""
    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY")
    print("=" * 90)
    
    print(f"\n{'Architecture':<20} {'Params':>12} {'Time (s)':>12} {'Final Reward':>14} {'Best Reward':>12} {'Ep/s':>8}")
    print("-" * 90)
    
    for arch, metrics in results['metrics'].items():
        name = arch_names.get(arch, arch)
        print(f"{name:<20} {metrics['num_params']:>12,} {metrics['avg_time']:>12.1f} "
              f"{metrics['avg_final_reward']:>14.1f} {metrics['avg_best_reward']:>12.1f} "
              f"{metrics['episodes_per_second']:>8.1f}")
    
    print("=" * 90)
    
    # Find best performers
    best_performance = max(results['metrics'].items(), key=lambda x: x[1]['avg_final_reward'])
    fastest = max(results['metrics'].items(), key=lambda x: x[1]['episodes_per_second'])
    smallest = min(results['metrics'].items(), key=lambda x: x[1]['num_params'])
    
    print(f"\n🏆 Best Performance: {arch_names.get(best_performance[0], best_performance[0])} "
          f"(Final Reward: {best_performance[1]['avg_final_reward']:.1f})")
    print(f"⚡ Fastest Training: {arch_names.get(fastest[0], fastest[0])} "
          f"({fastest[1]['episodes_per_second']:.1f} ep/s)")
    print(f"📦 Smallest Model: {arch_names.get(smallest[0], smallest[0])} "
          f"({smallest[1]['num_params']:,} params)")
    print("=" * 90 + "\n")


def _generate_summary_dict(results: Dict, arch_names: Dict) -> Dict:
    """Generate a summary dictionary of comparison results."""
    best_performance = max(results['metrics'].items(), key=lambda x: x[1]['avg_final_reward'])
    fastest = max(results['metrics'].items(), key=lambda x: x[1]['episodes_per_second'])
    smallest = min(results['metrics'].items(), key=lambda x: x[1]['num_params'])
    
    return {
        'best_performance': {
            'architecture': best_performance[0],
            'name': arch_names.get(best_performance[0], best_performance[0]),
            'final_reward': best_performance[1]['avg_final_reward']
        },
        'fastest': {
            'architecture': fastest[0],
            'name': arch_names.get(fastest[0], fastest[0]),
            'episodes_per_second': fastest[1]['episodes_per_second']
        },
        'smallest': {
            'architecture': smallest[0],
            'name': arch_names.get(smallest[0], smallest[0]),
            'num_params': smallest[1]['num_params']
        }
    }


# ============================================================================
# 8. MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main training demonstration with architecture comparison."""
    print("\n" + "=" * 70)
    print("FLAX NNX: DEEP Q-NETWORK (DQN) REINFORCEMENT LEARNING")
    print("=" * 70)
    
    # Check for gymnax availability and demonstrate if available
    if HAS_GYMNAX:
        print("\n🚀 gymnax detected - demonstrating JAX-native training...")
        print("=" * 70)
        
        # Train with gymnax (faster, JAX-native)
        print("\n📊 Training DQN with gymnax (JAX-native environment)...")
        agent_gymnax, rewards_gymnax, metrics_gymnax = train_dqn_gymnax(
            num_episodes=100,
            architecture='standard',
            hidden_dim=128,
            seed=42,
            verbose=True
        )
        
        # Evaluate with gymnax
        print("\n🎯 Evaluating gymnax-trained agent...")
        eval_reward_gymnax = evaluate_agent_gymnax(agent_gymnax, num_episodes=10)
        print(f"Average evaluation reward (gymnax): {eval_reward_gymnax:.1f}")
        
        # Demonstrate JAX-native rollout
        print("\n🔧 Demonstrating JAX-native rollout with jax.lax.scan...")
        key = jax.random.key(42)
        
        # Create a simple policy function for demonstration
        def random_policy(params, obs, key):
            """Random policy for demonstration."""
            return jax.random.randint(key, (), 0, 2)
        
        obs, actions, rewards, next_obs, dones = gymnax_rollout(
            key, random_policy, None, 200, "CartPole-v1"
        )
        print(f"  Rollout shapes - obs: {obs.shape}, rewards: {rewards.shape}")
        print(f"  Total reward: {jnp.sum(rewards):.1f}")
        
        # Demonstrate parallel rollouts with vmap
        print("\n🔧 Demonstrating parallel rollouts with vmap...")
        keys = jax.random.split(key, 8)  # 8 parallel environments
        obs_batch, _, rewards_batch, _, _ = gymnax_vmap_rollout(
            keys, random_policy, None, 200, "CartPole-v1"
        )
        print(f"  Parallel rollout shapes - obs: {obs_batch.shape}")
        print(f"  Mean total reward across envs: {jnp.mean(jnp.sum(rewards_batch, axis=1)):.1f}")
        
        print("\n" + "=" * 70)
    else:
        print("\n⚠️ gymnax not installed - using gymnasium only")
        print("   Install gymnax: pip install gymnax")
    
    # Compare all architectures (using gymnasium for full comparison)
    print("\n📊 Comparing DQN Architectures...")
    comparison_results = compare_architectures(
        architectures=['standard', 'dueling', 'noisy', 'deep'],
        num_episodes=150,
        num_runs=1,
        hidden_dim=128,
        seed=42,
        save_dir="comparison_plots"
    )
    
    # Train best performer for video recording
    best_arch = comparison_results['summary']['best_performance']['architecture']
    print(f"\n🎯 Training best performer ({best_arch}) for evaluation and video...")
    
    agent, rewards, metrics = train_dqn(
        num_episodes=200,
        architecture=best_arch,
        hidden_dim=128,
        seed=42,
        verbose=True
    )
    
    # Evaluate
    print("\n🎯 Evaluating trained agent...")
    eval_reward = evaluate_agent(agent, num_episodes=10)
    print(f"Average evaluation reward: {eval_reward:.1f}")
    
    # Plot training progress for best model
    print("\n📈 Plotting training progress...")
    plot_training_progress(
        rewards,
        save_path=f"comparison_plots/{best_arch}_training_progress.png",
        title=f"{best_arch.title()} DQN Training Progress"
    )
    
    # Record video of trained agent
    print("\n🎬 Recording video of trained agent...")
    try:
        video_path = record_video(
            agent,
            video_folder="videos",
            video_name=f"{best_arch}_dqn_cartpole",
            num_episodes=3,
            seed=42
        )
    except Exception as e:
        print(f"Video recording skipped: {e}")
        print("Note: Video recording requires 'moviepy' package and may not work in headless environments.")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBest Architecture: {comparison_results['summary']['best_performance']['name']}")
    print(f"Final Eval Reward: {eval_reward:.1f}")
    print("\nKey Components Demonstrated:")
    print("  ✓ Standard Q-Network architecture")
    print("  ✓ Dueling DQN architecture")
    print("  ✓ NoisyNet DQN architecture")
    print("  ✓ Deep DQN (4-layer) architecture")
    print("  ✓ Architecture comparison (performance, speed, size, weights)")
    print("  ✓ Experience replay buffer")
    print("  ✓ Epsilon-greedy exploration")
    print("  ✓ Target network with soft updates")
    print("  ✓ TD-learning training loop")
    print("  ✓ Video recording of trained agent")
    print("  ✓ Training progress visualization")
    if HAS_GYMNAX:
        print("  ✓ gymnax JAX-native environment integration")
        print("  ✓ JAX-native rollouts with jax.lax.scan")
        print("  ✓ Parallel rollouts with vmap")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
