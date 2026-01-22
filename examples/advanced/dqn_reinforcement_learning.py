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
- Integration with gymnasium for RL environments
- Video recording of trained agent simulation

Run: python advanced/dqn_reinforcement_learning.py

Reference:
    Mnih et al. "Playing Atari with Deep Reinforcement Learning"
    DeepMind 2013. https://arxiv.org/abs/1312.5602
    
    Mnih et al. "Human-level control through deep reinforcement learning"
    Nature 2015. https://doi.org/10.1038/nature14236
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from typing import Dict, List, Tuple, NamedTuple, Optional
import time
from collections import deque
import random
import gymnasium as gym

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
            use_dueling: Whether to use dueling architecture
            seed: Random seed
        """
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


# ============================================================================
# 6. TRAINING LOOP
# ============================================================================

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
    eval_frequency: int = 20,
    seed: int = 42
) -> Tuple[DQNAgent, List[float]]:
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
        use_dueling: Whether to use dueling architecture
        eval_frequency: Episodes between evaluations
        seed: Random seed
        
    Returns:
        Trained agent and episode rewards
    """
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
    best_reward = 0
    
    print(f"\n{'='*70}")
    print("DEEP Q-NETWORK (DQN) TRAINING")
    print(f"{'='*70}")
    print(f"Architecture: {'Dueling DQN' if use_dueling else 'Standard DQN'}")
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
        
        # Update best reward
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # Logging
        if episode % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Best: {best_reward:6.1f} | "
                  f"Epsilon: {agent.exploration.epsilon:.3f} | "
                  f"Buffer: {len(agent.replay_buffer):6d} | "
                  f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Best episode reward: {best_reward:.1f}")
    print(f"Final average (last 20): {np.mean(episode_rewards[-20:]):.1f}")
    print(f"{'='*70}\n")
    
    env.close()
    return agent, episode_rewards


def evaluate_agent(agent: DQNAgent, num_episodes: int = 10, seed: int = 0) -> float:
    """
    Evaluate trained agent.
    
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


# ============================================================================
# 8. MAIN DEMONSTRATION
# ============================================================================

def main():
    """Main training demonstration."""
    print("\n" + "=" * 70)
    print("FLAX NNX: DEEP Q-NETWORK (DQN) REINFORCEMENT LEARNING")
    print("=" * 70)
    
    # Train with standard DQN
    print("\n📊 Training Standard DQN...")
    agent, rewards = train_dqn(
        num_episodes=200,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
        epsilon_decay_steps=3000,
        hidden_dim=128,
        use_dueling=False,
        eval_frequency=20,
        seed=42
    )
    
    # Evaluate
    print("\n🎯 Evaluating trained agent...")
    eval_reward = evaluate_agent(agent, num_episodes=10)
    print(f"Average evaluation reward: {eval_reward:.1f}")
    
    # Train with Dueling DQN
    print("\n📊 Training Dueling DQN...")
    dueling_agent, dueling_rewards = train_dqn(
        num_episodes=200,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=10000,
        epsilon_decay_steps=3000,
        hidden_dim=128,
        use_dueling=True,
        eval_frequency=20,
        seed=42
    )
    
    # Evaluate dueling
    print("\n🎯 Evaluating Dueling DQN agent...")
    dueling_eval_reward = evaluate_agent(dueling_agent, num_episodes=10)
    print(f"Average evaluation reward (Dueling): {dueling_eval_reward:.1f}")
    
    # Plot training progress
    print("\n📈 Plotting training progress...")
    plot_training_progress(
        rewards, 
        save_path="dqn_training_progress.png",
        title="Standard DQN Training Progress"
    )
    plot_training_progress(
        dueling_rewards,
        save_path="dueling_dqn_training_progress.png", 
        title="Dueling DQN Training Progress"
    )
    
    # Record video of trained agent
    print("\n🎬 Recording video of trained Dueling DQN agent...")
    try:
        video_path = record_video(
            dueling_agent,
            video_folder="videos",
            video_name="dueling_dqn_cartpole",
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
    print(f"Standard DQN - Eval Reward: {eval_reward:.1f}")
    print(f"Dueling DQN  - Eval Reward: {dueling_eval_reward:.1f}")
    print("\nKey Components Demonstrated:")
    print("  ✓ Q-Network architecture with Flax NNX")
    print("  ✓ Dueling DQN architecture")
    print("  ✓ Experience replay buffer")
    print("  ✓ Epsilon-greedy exploration")
    print("  ✓ Target network with soft updates")
    print("  ✓ TD-learning training loop")
    print("  ✓ Video recording of trained agent")
    print("  ✓ Training progress visualization")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
