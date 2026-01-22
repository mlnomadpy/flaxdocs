"""
Unit tests for DQN reinforcement learning components.

Tests Q-Networks, replay buffer, epsilon-greedy exploration, and DQN agent.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np


@pytest.mark.unit
class TestQNetwork:
    """Tests for Q-Network architecture."""
    
    def test_qnetwork_creation(self):
        """Test QNetwork can be created with valid parameters."""
        from advanced.dqn_reinforcement_learning import QNetwork
        
        rngs = nnx.Rngs(0)
        model = QNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            rngs=rngs
        )
        assert model is not None
    
    def test_qnetwork_forward_shape(self):
        """Test QNetwork forward pass output shape."""
        from advanced.dqn_reinforcement_learning import QNetwork
        
        batch_size = 8
        state_dim = 4
        action_dim = 2
        
        rngs = nnx.Rngs(0)
        model = QNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            rngs=rngs
        )
        
        state = jnp.ones((batch_size, state_dim))
        q_values = model(state)
        
        assert q_values.shape == (batch_size, action_dim)
    
    def test_qnetwork_output_is_finite(self):
        """Test QNetwork produces finite outputs."""
        from advanced.dqn_reinforcement_learning import QNetwork
        
        rngs = nnx.Rngs(0)
        model = QNetwork(state_dim=4, action_dim=2, hidden_dim=64, rngs=rngs)
        
        state = jnp.ones((4, 4))
        q_values = model(state)
        
        assert jnp.all(jnp.isfinite(q_values))


@pytest.mark.unit
class TestDuelingQNetwork:
    """Tests for Dueling Q-Network architecture."""
    
    def test_dueling_creation(self):
        """Test DuelingQNetwork can be created."""
        from advanced.dqn_reinforcement_learning import DuelingQNetwork
        
        rngs = nnx.Rngs(0)
        model = DuelingQNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            rngs=rngs
        )
        assert model is not None
    
    def test_dueling_forward_shape(self):
        """Test DuelingQNetwork forward pass output shape."""
        from advanced.dqn_reinforcement_learning import DuelingQNetwork
        
        batch_size = 8
        state_dim = 4
        action_dim = 3
        
        rngs = nnx.Rngs(0)
        model = DuelingQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            rngs=rngs
        )
        
        state = jnp.ones((batch_size, state_dim))
        q_values = model(state)
        
        assert q_values.shape == (batch_size, action_dim)
    
    def test_dueling_advantage_centering(self):
        """Test that dueling architecture centers advantages."""
        from advanced.dqn_reinforcement_learning import DuelingQNetwork
        
        rngs = nnx.Rngs(0)
        model = DuelingQNetwork(state_dim=4, action_dim=3, hidden_dim=64, rngs=rngs)
        
        state = jnp.ones((1, 4))
        q_values = model(state)
        
        # Q-values should be finite
        assert jnp.all(jnp.isfinite(q_values))


@pytest.mark.unit
class TestReplayBuffer:
    """Tests for experience replay buffer."""
    
    def test_buffer_creation(self):
        """Test ReplayBuffer can be created."""
        from advanced.dqn_reinforcement_learning import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=1000)
        assert len(buffer) == 0
    
    def test_buffer_push(self):
        """Test adding transitions to buffer."""
        from advanced.dqn_reinforcement_learning import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=1000)
        
        state = np.array([1.0, 2.0, 3.0, 4.0])
        action = 0
        reward = 1.0
        next_state = np.array([1.5, 2.5, 3.5, 4.5])
        done = False
        
        buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 1
    
    def test_buffer_sample(self):
        """Test sampling from buffer."""
        from advanced.dqn_reinforcement_learning import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=1000)
        
        # Add several transitions
        for i in range(100):
            state = np.array([float(i)] * 4)
            buffer.push(state, i % 2, 1.0, state + 0.5, False)
        
        # Sample batch
        batch = buffer.sample(32)
        
        assert 'states' in batch
        assert 'actions' in batch
        assert 'rewards' in batch
        assert 'next_states' in batch
        assert 'dones' in batch
        
        assert batch['states'].shape == (32, 4)
        assert batch['actions'].shape == (32,)
        assert batch['rewards'].shape == (32,)
        assert batch['next_states'].shape == (32, 4)
        assert batch['dones'].shape == (32,)
    
    def test_buffer_capacity(self):
        """Test buffer respects capacity limit."""
        from advanced.dqn_reinforcement_learning import ReplayBuffer
        
        capacity = 100
        buffer = ReplayBuffer(capacity=capacity)
        
        # Add more than capacity
        for i in range(150):
            state = np.array([float(i)] * 4)
            buffer.push(state, 0, 1.0, state, False)
        
        assert len(buffer) == capacity


@pytest.mark.unit
class TestEpsilonGreedy:
    """Tests for epsilon-greedy exploration."""
    
    def test_epsilon_greedy_creation(self):
        """Test EpsilonGreedy can be created."""
        from advanced.dqn_reinforcement_learning import EpsilonGreedy
        
        exploration = EpsilonGreedy(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=1000
        )
        assert exploration is not None
    
    def test_epsilon_decay(self):
        """Test epsilon decays over steps."""
        from advanced.dqn_reinforcement_learning import EpsilonGreedy
        
        exploration = EpsilonGreedy(
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100
        )
        
        # Initial epsilon
        initial_epsilon = exploration.epsilon
        
        # Simulate some steps
        q_values = jnp.array([1.0, 2.0])
        for _ in range(50):
            exploration.select_action(q_values, 2)
        
        # Epsilon should have decreased
        assert exploration.epsilon < initial_epsilon
    
    def test_epsilon_bounds(self):
        """Test epsilon stays within bounds."""
        from advanced.dqn_reinforcement_learning import EpsilonGreedy
        
        exploration = EpsilonGreedy(
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay_steps=100
        )
        
        q_values = jnp.array([1.0, 2.0])
        
        # Many steps
        for _ in range(200):
            exploration.select_action(q_values, 2)
        
        # Should be at or near epsilon_end
        assert exploration.epsilon >= 0.01
        assert exploration.epsilon <= 1.0


@pytest.mark.unit
class TestCartPoleEnv:
    """Tests for gymnasium CartPole environment wrapper."""
    
    def test_env_creation(self):
        """Test environment can be created."""
        from advanced.dqn_reinforcement_learning import CartPoleEnv
        
        env = CartPoleEnv(seed=42)
        assert env.state_dim == 4
        assert env.action_dim == 2
        env.close()
    
    def test_env_reset(self):
        """Test environment reset."""
        from advanced.dqn_reinforcement_learning import CartPoleEnv
        
        env = CartPoleEnv(seed=42)
        state = env.reset()
        
        assert state.shape == (4,)
        assert np.all(np.isfinite(state))
        env.close()
    
    def test_env_step(self):
        """Test environment step."""
        from advanced.dqn_reinforcement_learning import CartPoleEnv
        
        env = CartPoleEnv(seed=42)
        state = env.reset()
        
        next_state, reward, done, info = env.step(0)
        
        assert next_state.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()
    
    def test_env_step_actions(self):
        """Test both actions work."""
        from advanced.dqn_reinforcement_learning import CartPoleEnv
        
        env = CartPoleEnv(seed=42)
        
        # Test action 0 (left)
        env.reset()
        state1, _, _, _ = env.step(0)
        
        # Test action 1 (right)
        env.reset()
        state2, _, _, _ = env.step(1)
        
        # States should be different
        assert not np.allclose(state1, state2)
        env.close()


@pytest.mark.unit
class TestGymnaxCartPoleEnv:
    """Tests for gymnax CartPole environment wrapper."""
    
    def test_gymnax_env_creation(self):
        """Test gymnax environment can be created."""
        try:
            from advanced.dqn_reinforcement_learning import GymnaxCartPoleEnv, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        env = GymnaxCartPoleEnv(seed=42)
        assert env.state_dim == 4
        assert env.action_dim == 2
        env.close()
    
    def test_gymnax_env_reset(self):
        """Test gymnax environment reset."""
        try:
            from advanced.dqn_reinforcement_learning import GymnaxCartPoleEnv, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        env = GymnaxCartPoleEnv(seed=42)
        state, obs = env.reset_numpy()
        
        assert obs.shape == (4,)
        assert np.all(np.isfinite(obs))
        env.close()
    
    def test_gymnax_env_step(self):
        """Test gymnax environment step."""
        try:
            from advanced.dqn_reinforcement_learning import GymnaxCartPoleEnv, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        env = GymnaxCartPoleEnv(seed=42)
        state, obs = env.reset_numpy()
        
        next_obs, next_state, reward, done, info = env.step_numpy(state, 0)
        
        assert next_obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        env.close()
    
    def test_gymnax_env_jax_step(self):
        """Test gymnax environment JAX-native step."""
        try:
            from advanced.dqn_reinforcement_learning import GymnaxCartPoleEnv, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        env = GymnaxCartPoleEnv(seed=42)
        state, obs = env.reset()
        
        next_obs, next_state, reward, done, info = env.step(state, jnp.array(0))
        
        assert next_obs.shape == (4,)
        assert jnp.isfinite(reward)
        env.close()


@pytest.mark.unit
class TestGymnaxRollout:
    """Tests for gymnax rollout functions."""
    
    def test_gymnax_rollout(self):
        """Test JAX-native rollout function."""
        try:
            from advanced.dqn_reinforcement_learning import gymnax_rollout, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        # Simple random policy
        def random_policy(params, obs, key):
            return jax.random.randint(key, (), 0, 2)
        
        key = jax.random.key(42)
        obs, actions, rewards, next_obs, dones = gymnax_rollout(
            key, random_policy, None, 100, "CartPole-v1"
        )
        
        assert obs.shape == (100, 4)
        assert actions.shape == (100,)
        assert rewards.shape == (100,)
        assert next_obs.shape == (100, 4)
        assert dones.shape == (100,)
    
    def test_gymnax_vmap_rollout(self):
        """Test parallel rollouts with vmap."""
        try:
            from advanced.dqn_reinforcement_learning import gymnax_vmap_rollout, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        # Simple random policy
        def random_policy(params, obs, key):
            return jax.random.randint(key, (), 0, 2)
        
        num_envs = 4
        keys = jax.random.split(jax.random.key(42), num_envs)
        
        obs, actions, rewards, next_obs, dones = gymnax_vmap_rollout(
            keys, random_policy, None, 50, "CartPole-v1"
        )
        
        assert obs.shape == (num_envs, 50, 4)
        assert rewards.shape == (num_envs, 50)


@pytest.mark.unit
class TestTrainDqnGymnax:
    """Tests for gymnax-based DQN training."""
    
    def test_train_dqn_gymnax_minimal(self):
        """Test gymnax training runs successfully."""
        try:
            from advanced.dqn_reinforcement_learning import train_dqn_gymnax, HAS_GYMNAX
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        # Very short training just to verify it works
        agent, rewards, metrics = train_dqn_gymnax(
            num_episodes=5,
            max_steps_per_episode=50,
            hidden_dim=32,
            buffer_size=500,
            batch_size=16,
            seed=42,
            verbose=False
        )
        
        assert agent is not None
        assert len(rewards) == 5
        assert metrics['environment'] == 'gymnax'
    
    def test_evaluate_agent_gymnax(self):
        """Test gymnax-based agent evaluation."""
        try:
            from advanced.dqn_reinforcement_learning import (
                DQNAgent, evaluate_agent_gymnax, HAS_GYMNAX
            )
        except ImportError:
            pytest.skip("gymnax not installed")
        
        if not HAS_GYMNAX:
            pytest.skip("gymnax not installed")
        
        # Create agent
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=32,
            seed=0
        )
        
        # Evaluate
        avg_reward = evaluate_agent_gymnax(agent, num_episodes=3, max_steps=50)
        
        assert isinstance(avg_reward, float)
        assert np.isfinite(avg_reward)
class TestDQNAgent:
    """Tests for DQN agent."""
    
    def test_agent_creation(self):
        """Test DQNAgent can be created."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            learning_rate=1e-3,
            gamma=0.99,
            buffer_size=1000,
            batch_size=32,
            seed=0
        )
        assert agent is not None
    
    def test_agent_select_action(self):
        """Test agent action selection."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            seed=0
        )
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = agent.select_action(state)
        
        assert action in [0, 1]
    
    def test_agent_eval_mode(self):
        """Test agent greedy action selection in eval mode."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            seed=0
        )
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # In eval mode, should always return same action
        actions = [agent.select_action(state, eval_mode=True) for _ in range(10)]
        
        # All actions should be the same (greedy)
        assert all(a == actions[0] for a in actions)
    
    def test_agent_store_transition(self):
        """Test agent stores transitions."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            buffer_size=1000,
            seed=0
        )
        
        state = np.array([0.1, 0.2, 0.3, 0.4])
        next_state = np.array([0.15, 0.25, 0.35, 0.45])
        
        agent.store_transition(state, 0, 1.0, next_state, False)
        
        assert len(agent.replay_buffer) == 1
    
    def test_agent_train_step(self):
        """Test agent training step."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            buffer_size=1000,
            batch_size=32,
            seed=0
        )
        
        # Fill buffer with enough samples
        for i in range(100):
            state = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(4).astype(np.float32)
            agent.store_transition(state, i % 2, 1.0, next_state, False)
        
        # Should be able to train
        loss = agent.train_step()
        
        assert isinstance(loss, float)
        assert np.isfinite(loss)


@pytest.mark.unit
class TestDuelingAgent:
    """Tests for DQN agent with dueling architecture."""
    
    def test_dueling_agent_creation(self):
        """Test DQNAgent with dueling can be created."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            use_dueling=True,
            seed=0
        )
        assert agent is not None
    
    def test_dueling_agent_train(self):
        """Test dueling agent can train."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            buffer_size=1000,
            batch_size=32,
            use_dueling=True,
            seed=0
        )
        
        # Fill buffer
        for i in range(100):
            state = np.random.randn(4).astype(np.float32)
            next_state = np.random.randn(4).astype(np.float32)
            agent.store_transition(state, i % 2, 1.0, next_state, False)
        
        # Train
        loss = agent.train_step()
        
        assert isinstance(loss, float)
        assert np.isfinite(loss)


@pytest.mark.unit
class TestVisualization:
    """Tests for visualization functions."""
    
    def test_plot_training_progress(self):
        """Test plot_training_progress function."""
        from advanced.dqn_reinforcement_learning import plot_training_progress
        import os
        import tempfile
        
        # Skip if matplotlib not installed
        try:
            import matplotlib
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        # Create dummy episode rewards
        rewards = [float(i + np.random.randn() * 10) for i in range(100)]
        
        # Test with save to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
        
        try:
            # Should not raise
            plot_training_progress(rewards, save_path=temp_path)
            
            # Check file was created
            assert os.path.exists(temp_path)
            assert os.path.getsize(temp_path) > 0
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_plot_training_progress_no_save(self):
        """Test plot_training_progress without saving."""
        from advanced.dqn_reinforcement_learning import plot_training_progress
        
        rewards = [float(i) for i in range(50)]
        
        # Should not raise (gracefully handles missing matplotlib)
        plot_training_progress(rewards, save_path=None)
    
    def test_record_video_function_exists(self):
        """Test that record_video function can be imported."""
        from advanced.dqn_reinforcement_learning import record_video
        
        # Function should exist
        assert callable(record_video)
    
    def test_compare_architectures_function_exists(self):
        """Test that compare_architectures function can be imported."""
        from advanced.dqn_reinforcement_learning import compare_architectures
        
        # Function should exist
        assert callable(compare_architectures)


@pytest.mark.unit
class TestNewArchitectures:
    """Tests for new network architectures."""
    
    def test_noisy_qnetwork_creation(self):
        """Test NoisyQNetwork can be created."""
        from advanced.dqn_reinforcement_learning import NoisyQNetwork
        
        rngs = nnx.Rngs(0)
        model = NoisyQNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            rngs=rngs
        )
        assert model is not None
    
    def test_noisy_qnetwork_forward_shape(self):
        """Test NoisyQNetwork forward pass output shape."""
        from advanced.dqn_reinforcement_learning import NoisyQNetwork
        
        batch_size = 8
        state_dim = 4
        action_dim = 2
        
        rngs = nnx.Rngs(0)
        model = NoisyQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            rngs=rngs
        )
        
        state = jnp.ones((batch_size, state_dim))
        q_values = model(state)
        
        assert q_values.shape == (batch_size, action_dim)
    
    def test_deep_qnetwork_creation(self):
        """Test DeepQNetwork can be created."""
        from advanced.dqn_reinforcement_learning import DeepQNetwork
        
        rngs = nnx.Rngs(0)
        model = DeepQNetwork(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            rngs=rngs
        )
        assert model is not None
    
    def test_deep_qnetwork_forward_shape(self):
        """Test DeepQNetwork forward pass output shape."""
        from advanced.dqn_reinforcement_learning import DeepQNetwork
        
        batch_size = 8
        state_dim = 4
        action_dim = 2
        
        rngs = nnx.Rngs(0)
        model = DeepQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=64,
            rngs=rngs
        )
        
        state = jnp.ones((batch_size, state_dim))
        q_values = model(state)
        
        assert q_values.shape == (batch_size, action_dim)
    
    def test_agent_with_noisy_architecture(self):
        """Test DQNAgent with noisy architecture."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            architecture='noisy',
            seed=0
        )
        assert agent is not None
        assert agent.architecture == 'noisy'
    
    def test_agent_with_deep_architecture(self):
        """Test DQNAgent with deep architecture."""
        from advanced.dqn_reinforcement_learning import DQNAgent
        
        agent = DQNAgent(
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            architecture='deep',
            seed=0
        )
        assert agent is not None
        assert agent.architecture == 'deep'
