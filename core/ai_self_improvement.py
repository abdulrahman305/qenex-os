#!/usr/bin/env python3
"""
QENEX AI Self-Improvement System
Autonomous learning and optimization for financial operations
"""

import asyncio
import json
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import pickle
from pathlib import Path
import threading
from collections import deque
import random

logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    TRANSACTION_SPEED = "transaction_speed"
    RISK_ACCURACY = "risk_accuracy"
    FRAUD_DETECTION = "fraud_detection"
    COST_EFFICIENCY = "cost_efficiency"
    LIQUIDITY_MANAGEMENT = "liquidity_management"
    COMPLIANCE_ACCURACY = "compliance_accuracy"

@dataclass
class PerformanceMetric:
    metric_name: str
    current_value: float
    target_value: float
    improvement_rate: float
    last_updated: datetime
    history: List[float] = field(default_factory=list)
    
    def update(self, new_value: float):
        self.history.append(self.current_value)
        if len(self.history) > 1000:
            self.history.pop(0)
        self.current_value = new_value
        self.last_updated = datetime.now()
        self.improvement_rate = self.calculate_improvement_rate()
    
    def calculate_improvement_rate(self) -> float:
        if len(self.history) < 2:
            return 0.0
        recent = self.history[-min(10, len(self.history)):]
        if recent[0] == 0:
            return 0.0
        return ((self.current_value - recent[0]) / recent[0]) * 100

@dataclass
class LearningModel:
    model_id: str
    model_type: str
    version: float
    accuracy: float
    parameters: Dict[str, Any]
    training_data_size: int
    last_trained: datetime
    performance_history: List[float] = field(default_factory=list)
    
    def evaluate_performance(self) -> float:
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history[-10:]) / min(10, len(self.performance_history))

class NeuralOptimizer:
    """Neural network-based optimization engine"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize weights with Xavier initialization
        self.weights = {
            'W1': np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim),
            'b2': np.zeros((1, hidden_dim)),
            'W3': np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
            'b3': np.zeros((1, output_dim))
        }
        
        # Adam optimizer parameters
        self.adam_params = {
            'm': {k: np.zeros_like(v) for k, v in self.weights.items()},
            'v': {k: np.zeros_like(v) for k, v in self.weights.items()},
            't': 0,
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8
        }
        
        self.training_history = []
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Forward propagation with caching for backprop"""
        cache = {}
        
        # Layer 1
        cache['Z1'] = X @ self.weights['W1'] + self.weights['b1']
        cache['A1'] = self.relu(cache['Z1'])
        
        # Layer 2
        cache['Z2'] = cache['A1'] @ self.weights['W2'] + self.weights['b2']
        cache['A2'] = self.relu(cache['Z2'])
        
        # Output layer
        cache['Z3'] = cache['A2'] @ self.weights['W3'] + self.weights['b3']
        output = self.sigmoid(cache['Z3'])
        
        return output, cache
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                 cache: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Backward propagation"""
        m = X.shape[0]
        gradients = {}
        
        # Output layer gradients
        dZ3 = output - y
        gradients['W3'] = (cache['A2'].T @ dZ3) / m
        gradients['b3'] = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Layer 2 gradients
        dA2 = dZ3 @ self.weights['W3'].T
        dZ2 = dA2 * self.relu_derivative(cache['Z2'])
        gradients['W2'] = (cache['A1'].T @ dZ2) / m
        gradients['b2'] = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Layer 1 gradients
        dA1 = dZ2 @ self.weights['W2'].T
        dZ1 = dA1 * self.relu_derivative(cache['Z1'])
        gradients['W1'] = (X.T @ dZ1) / m
        gradients['b1'] = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return gradients
    
    def adam_update(self, gradients: Dict[str, np.ndarray]):
        """Adam optimizer update"""
        self.adam_params['t'] += 1
        t = self.adam_params['t']
        lr = self.adam_params['lr']
        beta1 = self.adam_params['beta1']
        beta2 = self.adam_params['beta2']
        epsilon = self.adam_params['epsilon']
        
        for key in self.weights:
            # Update biased first moment estimate
            self.adam_params['m'][key] = beta1 * self.adam_params['m'][key] + (1 - beta1) * gradients[key]
            
            # Update biased second raw moment estimate
            self.adam_params['v'][key] = beta2 * self.adam_params['v'][key] + (1 - beta2) * (gradients[key] ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.adam_params['m'][key] / (1 - beta1 ** t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.adam_params['v'][key] / (1 - beta2 ** t)
            
            # Update weights
            self.weights[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              batch_size: int = 32) -> List[float]:
        """Train the neural network"""
        losses = []
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output, cache = self.forward(X_batch)
                
                # Compute loss
                loss = self.binary_cross_entropy(y_batch, output)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                gradients = self.backward(X_batch, y_batch, output, cache)
                
                # Update weights
                self.adam_update(gradients)
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        self.training_history.extend(losses)
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        output, _ = self.forward(X)
        return output
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return (x > 0).astype(float)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class ReinforcementLearner:
    """Q-learning based optimization for decision making"""
    
    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = deque(maxlen=10000)
    
    def get_state_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key"""
        return hashlib.md5(state.tobytes()).hexdigest()
    
    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        state_key = self.get_state_key(state)
        
        if np.random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        
        return np.argmax(self.q_table[state_key])
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size: int = 32):
        """Experience replay for training"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state_key = self.get_state_key(state)
            next_state_key = self.get_state_key(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            
            target = reward
            if not done:
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.zeros(self.action_size)
                target = reward + self.discount_factor * np.max(self.q_table[next_state_key])
            
            self.q_table[state_key][action] = (
                (1 - self.learning_rate) * self.q_table[state_key][action] +
                self.learning_rate * target
            )
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class SelfImprovementEngine:
    """Main AI self-improvement and optimization system"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("./ai_models")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics
        self.metrics = {
            OptimizationTarget.TRANSACTION_SPEED: PerformanceMetric(
                "transaction_speed", 100.0, 10.0, 0.0, datetime.now()
            ),
            OptimizationTarget.RISK_ACCURACY: PerformanceMetric(
                "risk_accuracy", 0.85, 0.99, 0.0, datetime.now()
            ),
            OptimizationTarget.FRAUD_DETECTION: PerformanceMetric(
                "fraud_detection", 0.90, 0.995, 0.0, datetime.now()
            ),
            OptimizationTarget.COST_EFFICIENCY: PerformanceMetric(
                "cost_efficiency", 0.70, 0.95, 0.0, datetime.now()
            ),
            OptimizationTarget.LIQUIDITY_MANAGEMENT: PerformanceMetric(
                "liquidity_management", 0.80, 0.98, 0.0, datetime.now()
            ),
            OptimizationTarget.COMPLIANCE_ACCURACY: PerformanceMetric(
                "compliance_accuracy", 0.95, 0.999, 0.0, datetime.now()
            )
        }
        
        # Learning models
        self.models = {
            'risk_predictor': NeuralOptimizer(input_dim=20, hidden_dim=128, output_dim=1),
            'fraud_detector': NeuralOptimizer(input_dim=30, hidden_dim=256, output_dim=1),
            'liquidity_optimizer': NeuralOptimizer(input_dim=15, hidden_dim=64, output_dim=5),
            'decision_maker': ReinforcementLearner(state_size=50, action_size=10)
        }
        
        # Model metadata
        self.model_info = {
            'risk_predictor': LearningModel(
                'risk_predictor', 'neural_network', 1.0, 0.85, 
                {}, 0, datetime.now()
            ),
            'fraud_detector': LearningModel(
                'fraud_detector', 'neural_network', 1.0, 0.90,
                {}, 0, datetime.now()
            ),
            'liquidity_optimizer': LearningModel(
                'liquidity_optimizer', 'neural_network', 1.0, 0.80,
                {}, 0, datetime.now()
            ),
            'decision_maker': LearningModel(
                'decision_maker', 'reinforcement_learning', 1.0, 0.75,
                {}, 0, datetime.now()
            )
        }
        
        # Optimization strategies
        self.strategies = {}
        self.active_experiments = []
        
        # Threading for continuous learning
        self.learning_thread = None
        self.stop_learning = threading.Event()
        
        logger.info("Self-Improvement Engine initialized")
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze current system performance"""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {},
            'recommendations': [],
            'optimization_opportunities': []
        }
        
        for target, metric in self.metrics.items():
            performance_gap = metric.target_value - metric.current_value
            improvement_needed = performance_gap > 0
            
            analysis['metrics'][target.value] = {
                'current': metric.current_value,
                'target': metric.target_value,
                'gap': performance_gap,
                'improvement_rate': metric.improvement_rate,
                'needs_improvement': improvement_needed
            }
            
            if improvement_needed:
                analysis['optimization_opportunities'].append({
                    'target': target.value,
                    'priority': self._calculate_priority(metric),
                    'estimated_effort': self._estimate_effort(metric)
                })
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis['optimization_opportunities'])
        
        return analysis
    
    def _calculate_priority(self, metric: PerformanceMetric) -> str:
        """Calculate optimization priority"""
        gap_percentage = abs(metric.target_value - metric.current_value) / metric.target_value * 100
        
        if gap_percentage > 50:
            return "CRITICAL"
        elif gap_percentage > 25:
            return "HIGH"
        elif gap_percentage > 10:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _estimate_effort(self, metric: PerformanceMetric) -> str:
        """Estimate effort required for improvement"""
        if metric.improvement_rate < 0:
            return "HIGH"
        elif metric.improvement_rate < 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        for opp in sorted(opportunities, key=lambda x: x['priority'] == "CRITICAL", reverse=True):
            if opp['target'] == 'transaction_speed':
                recommendations.append(
                    "Implement batch processing and parallel execution for transactions"
                )
            elif opp['target'] == 'risk_accuracy':
                recommendations.append(
                    "Enhance risk model with additional features and deep learning"
                )
            elif opp['target'] == 'fraud_detection':
                recommendations.append(
                    "Deploy anomaly detection algorithms and pattern recognition"
                )
            elif opp['target'] == 'cost_efficiency':
                recommendations.append(
                    "Optimize resource allocation and implement auto-scaling"
                )
            elif opp['target'] == 'liquidity_management':
                recommendations.append(
                    "Implement predictive liquidity models and smart routing"
                )
            elif opp['target'] == 'compliance_accuracy':
                recommendations.append(
                    "Enhance rule engine and implement automated compliance checks"
                )
        
        return recommendations[:5]  # Top 5 recommendations
    
    async def optimize_model(self, model_name: str, training_data: np.ndarray, 
                            labels: np.ndarray) -> Dict[str, Any]:
        """Optimize a specific model with new data"""
        if model_name not in self.models:
            return {'error': f'Model {model_name} not found'}
        
        model = self.models[model_name]
        model_info = self.model_info[model_name]
        
        # Train the model
        if isinstance(model, NeuralOptimizer):
            losses = model.train(training_data, labels, epochs=50)
            
            # Evaluate performance
            predictions = model.predict(training_data)
            accuracy = self._calculate_accuracy(labels, predictions)
            
            # Update model info
            model_info.accuracy = accuracy
            model_info.training_data_size += len(training_data)
            model_info.last_trained = datetime.now()
            model_info.performance_history.append(accuracy)
            
            # Save model if improved
            if accuracy > max(model_info.performance_history[:-1], default=0):
                self._save_model(model_name, model)
                logger.info(f"Model {model_name} improved to {accuracy:.4f} accuracy")
            
            return {
                'model': model_name,
                'accuracy': accuracy,
                'improvement': accuracy - model_info.performance_history[-2] if len(model_info.performance_history) > 1 else 0,
                'training_samples': len(training_data),
                'final_loss': losses[-1] if losses else 0
            }
        
        return {'error': 'Model type not supported for optimization'}
    
    def _calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate prediction accuracy"""
        if y_pred.shape != y_true.shape:
            return 0.0
        
        # For binary classification
        if y_pred.shape[1] == 1:
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        
        # For multi-class
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(predictions == true_labels)
    
    def _save_model(self, model_name: str, model: Any):
        """Save model to disk"""
        model_path = self.data_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    def _load_model(self, model_name: str) -> Optional[Any]:
        """Load model from disk"""
        model_path = self.data_dir / f"{model_name}.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    async def run_experiment(self, experiment_name: str, 
                            hypothesis: str, 
                            test_function: Callable) -> Dict[str, Any]:
        """Run an optimization experiment"""
        experiment = {
            'name': experiment_name,
            'hypothesis': hypothesis,
            'start_time': datetime.now(),
            'status': 'RUNNING',
            'results': {}
        }
        
        self.active_experiments.append(experiment)
        
        try:
            # Run the test function
            results = await test_function()
            
            experiment['results'] = results
            experiment['status'] = 'COMPLETED'
            experiment['end_time'] = datetime.now()
            experiment['duration'] = (experiment['end_time'] - experiment['start_time']).total_seconds()
            
            # Analyze results
            experiment['success'] = self._evaluate_experiment(results)
            
            if experiment['success']:
                logger.info(f"Experiment {experiment_name} succeeded")
                # Apply successful optimizations
                await self._apply_optimization(experiment)
            
        except Exception as e:
            experiment['status'] = 'FAILED'
            experiment['error'] = str(e)
            logger.error(f"Experiment {experiment_name} failed: {e}")
        
        return experiment
    
    def _evaluate_experiment(self, results: Dict[str, Any]) -> bool:
        """Evaluate if experiment was successful"""
        # Check if key metrics improved
        if 'metrics' in results:
            improvements = sum(1 for v in results['metrics'].values() if v.get('improved', False))
            return improvements > len(results['metrics']) / 2
        return False
    
    async def _apply_optimization(self, experiment: Dict[str, Any]):
        """Apply successful optimization to the system"""
        # This would integrate the successful optimization into production
        logger.info(f"Applying optimization from experiment: {experiment['name']}")
        
        # Update relevant metrics
        if 'metric_updates' in experiment['results']:
            for target, value in experiment['results']['metric_updates'].items():
                if target in self.metrics:
                    self.metrics[target].update(value)
    
    def start_continuous_learning(self):
        """Start continuous learning thread"""
        if self.learning_thread is None or not self.learning_thread.is_alive():
            self.stop_learning.clear()
            self.learning_thread = threading.Thread(target=self._learning_loop)
            self.learning_thread.start()
            logger.info("Continuous learning started")
    
    def stop_continuous_learning(self):
        """Stop continuous learning"""
        self.stop_learning.set()
        if self.learning_thread:
            self.learning_thread.join()
        logger.info("Continuous learning stopped")
    
    def _learning_loop(self):
        """Main learning loop running in background"""
        while not self.stop_learning.is_set():
            try:
                # Simulate learning cycle
                for model_name in self.models:
                    if self.stop_learning.is_set():
                        break
                    
                    # Generate synthetic training data (in production, this would be real data)
                    if model_name == 'risk_predictor':
                        X = np.random.randn(100, 20)
                        y = (np.sum(X, axis=1) > 0).astype(float).reshape(-1, 1)
                    elif model_name == 'fraud_detector':
                        X = np.random.randn(100, 30)
                        y = (np.sum(X[:, :5], axis=1) > 2).astype(float).reshape(-1, 1)
                    elif model_name == 'liquidity_optimizer':
                        X = np.random.randn(100, 15)
                        y = np.random.randn(100, 5)
                    else:
                        continue
                    
                    # Train model
                    if isinstance(self.models[model_name], NeuralOptimizer):
                        self.models[model_name].train(X, y, epochs=10)
                
                # Sleep before next iteration
                self.stop_learning.wait(60)  # Train every minute
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'models': {
                name: {
                    'accuracy': info.accuracy,
                    'version': info.version,
                    'last_trained': info.last_trained.isoformat(),
                    'performance': info.evaluate_performance()
                }
                for name, info in self.model_info.items()
            },
            'metrics': {
                target.value: {
                    'current': metric.current_value,
                    'target': metric.target_value,
                    'improvement_rate': metric.improvement_rate
                }
                for target, metric in self.metrics.items()
            },
            'active_experiments': len(self.active_experiments),
            'learning_active': self.learning_thread is not None and self.learning_thread.is_alive()
        }

async def main():
    """Test the self-improvement engine"""
    engine = SelfImprovementEngine()
    
    # Start continuous learning
    engine.start_continuous_learning()
    
    # Analyze performance
    analysis = await engine.analyze_performance()
    print(f"Performance Analysis: {json.dumps(analysis, indent=2)}")
    
    # Run an experiment
    async def test_optimization():
        # Simulate optimization test
        return {
            'metrics': {
                'speed': {'improved': True, 'value': 50},
                'accuracy': {'improved': True, 'value': 0.92}
            },
            'metric_updates': {
                OptimizationTarget.TRANSACTION_SPEED: 50
            }
        }
    
    experiment = await engine.run_experiment(
        "Speed Optimization Test",
        "Parallel processing will improve transaction speed by 50%",
        test_optimization
    )
    print(f"Experiment Result: {experiment}")
    
    # Get status
    status = engine.get_optimization_status()
    print(f"Optimization Status: {json.dumps(status, indent=2, default=str)}")
    
    # Stop learning
    await asyncio.sleep(5)
    engine.stop_continuous_learning()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())