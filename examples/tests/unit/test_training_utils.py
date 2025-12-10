"""
Unit tests for shared training utilities.

Tests common training functions like train_step, eval_step, and metrics.
"""

import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import optax


@pytest.mark.unit
class TestTrainingStep:
    """Tests for training step function."""
    
    def test_train_step_reduces_loss(self):
        """Test that train_step can reduce loss over iterations."""
        from shared.models import MLP
        from shared.training_utils import create_train_step
        
        # Simple regression problem
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=1, n_layers=2, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
        
        # Create training data
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 10))
        y = jax.random.normal(key, (32, 1))
        
        # Get train step function
        train_step = create_train_step(loss_fn_name='mse')
        
        # First loss
        loss1, _ = train_step(model, optimizer, {'x': x, 'y': y})
        
        # Train for a few steps
        for _ in range(10):
            loss, metrics = train_step(model, optimizer, {'x': x, 'y': y})
        
        # Loss should decrease
        assert loss < loss1
    
    def test_train_step_returns_metrics(self):
        """Test that train_step returns metrics dictionary."""
        from shared.models import MLP
        from shared.training_utils import create_train_step
        
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=1, n_layers=2, rngs=rngs)
        optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 10))
        y = jax.random.normal(key, (32, 1))
        
        train_step = create_train_step(loss_fn_name='mse')
        loss, metrics = train_step(model, optimizer, {'x': x, 'y': y})
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics


@pytest.mark.unit
class TestEvalStep:
    """Tests for evaluation step function."""
    
    def test_eval_step_no_gradient(self):
        """Test that eval step doesn't compute gradients."""
        from shared.models import MLP
        from shared.training_utils import create_eval_step
        
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=10, n_layers=2, rngs=rngs)
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 10))
        y = jax.random.randint(key, (32,), 0, 10)
        
        eval_step = create_eval_step(loss_fn_name='cross_entropy')
        
        # Should not raise an error
        loss, metrics = eval_step(model, {'x': x, 'y': y})
        
        assert isinstance(loss, (float, jax.Array))
        assert isinstance(metrics, dict)
    
    def test_eval_step_returns_accuracy(self):
        """Test that eval step returns accuracy for classification."""
        from shared.models import MLP
        from shared.training_utils import create_eval_step
        
        rngs = nnx.Rngs(0)
        model = MLP(in_features=10, hidden_features=20, out_features=10, n_layers=2, rngs=rngs)
        
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (32, 10))
        y = jax.random.randint(key, (32,), 0, 10)
        
        eval_step = create_eval_step(loss_fn_name='cross_entropy')
        loss, metrics = eval_step(model, {'x': x, 'y': y})
        
        assert 'accuracy' in metrics
        assert 0.0 <= metrics['accuracy'] <= 1.0


@pytest.mark.unit
class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss computation."""
        from shared.training_utils import compute_mse_loss
        
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        
        loss = compute_mse_loss(predictions, targets)
        
        expected_loss = jnp.mean((predictions - targets) ** 2)
        assert jnp.allclose(loss, expected_loss)
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss computation."""
        from shared.training_utils import compute_cross_entropy_loss
        
        logits = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]])
        labels = jnp.array([1, 2])
        
        loss = compute_cross_entropy_loss(logits, labels)
        
        assert jnp.isfinite(loss)
        assert loss > 0


@pytest.mark.unit
class TestMetrics:
    """Tests for metrics computation."""
    
    def test_accuracy_metric(self):
        """Test accuracy computation."""
        from shared.training_utils import compute_accuracy
        
        logits = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 2.0], [2.0, 0.5, 1.0]])
        labels = jnp.array([1, 2, 0])
        
        accuracy = compute_accuracy(logits, labels)
        
        # All predictions are correct
        assert accuracy == 1.0
    
    def test_accuracy_metric_partial(self):
        """Test accuracy with some incorrect predictions."""
        from shared.training_utils import compute_accuracy
        
        logits = jnp.array([[2.0, 1.0, 0.5], [0.5, 1.0, 2.0]])
        labels = jnp.array([1, 2])  # First is wrong (predicts 0), second is right
        
        accuracy = compute_accuracy(logits, labels)
        
        assert accuracy == 0.5


@pytest.mark.unit
class TestLearningRateSchedules:
    """Tests for learning rate schedules."""
    
    def test_warmup_cosine_schedule(self):
        """Test warmup + cosine decay schedule."""
        from shared.training_utils import create_warmup_cosine_schedule
        
        schedule = create_warmup_cosine_schedule(
            init_value=0.0,
            peak_value=1.0,
            warmup_steps=100,
            decay_steps=1000,
            end_value=0.0
        )
        
        # At step 0, should be init_value
        assert jnp.allclose(schedule(0), 0.0)
        
        # At warmup end, should be peak_value
        assert jnp.allclose(schedule(100), 1.0, atol=0.01)
        
        # After warmup, should decay
        assert schedule(500) < 1.0
        assert schedule(500) > 0.0
        
        # At end, should be end_value
        assert jnp.allclose(schedule(1100), 0.0, atol=0.01)
