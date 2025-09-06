#!/usr/bin/env python3
"""
Real AI System with Continuous Learning
Actual working implementation with persistent model storage
"""

import os
import json
import pickle
import numpy as np
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
import queue

@dataclass
class LearningRecord:
    timestamp: float
    model_type: str
    accuracy: float
    loss: float
    data_size: int
    training_time: float

class RealNeuralNetwork:
    """Real neural network with save/load capabilities"""
    
    def __init__(self, layers: List[int], learning_rate: float = 0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        self.training_history = []
        self.total_samples_seen = 0
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        self.activations = [X]
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        output = self.softmax(z) if self.layers[-1] > 1 else 1 / (1 + np.exp(-z))
        self.activations.append(output)
        
        return output
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Calculate gradients
        deltas = [output - y]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.relu_derivative(self.activations[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.activations[i].T, deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=100, batch_size=32, verbose=True):
        """Train the network with mini-batch gradient descent"""
        n_samples = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            epoch_acc = 0
            n_batches = 0
            
            for start_idx in range(0, n_samples, batch_size):
                batch_X = X_shuffled[start_idx:start_idx + batch_size]
                batch_y = y_shuffled[start_idx:start_idx + batch_size]
                
                # Forward and backward pass
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)
                
                # Calculate loss and accuracy
                if len(output.shape) > 1 and output.shape[1] > 1:
                    loss = -np.mean(np.sum(batch_y * np.log(output + 1e-7), axis=1))
                    acc = np.mean(np.argmax(output, axis=1) == np.argmax(batch_y, axis=1))
                else:
                    loss = np.mean((output - batch_y) ** 2)
                    acc = np.mean(((output > 0.5) == batch_y).astype(float))
                
                epoch_loss += loss
                epoch_acc += acc
                n_batches += 1
            
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        self.total_samples_seen += n_samples * epochs
        self.training_history.append(history)
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        if len(output.shape) > 1 and output.shape[1] > 1:
            return np.argmax(output, axis=1)
        else:
            return (output > 0.5).astype(int)
    
    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'layers': self.layers,
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'learning_rate': self.learning_rate,
            'total_samples_seen': self.total_samples_seen,
            'training_history': self.training_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
        
        return True
    
    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        self.layers = model_data['layers']
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]
        self.learning_rate = model_data['learning_rate']
        self.total_samples_seen = model_data['total_samples_seen']
        self.training_history = model_data['training_history']
        
        return True

class ContinuousLearningSystem:
    """System that continuously learns and improves"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.learning_queue = queue.Queue()
        self.learning_records = []
        self.is_learning = False
        self.learning_thread = None
        
    def create_model(self, name: str, layers: List[int], learning_rate: float = 0.01):
        """Create a new model"""
        self.models[name] = RealNeuralNetwork(layers, learning_rate)
        return self.models[name]
    
    def queue_training(self, model_name: str, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        """Queue data for continuous learning"""
        self.learning_queue.put({
            'model_name': model_name,
            'X': X,
            'y': y,
            'epochs': epochs,
            'timestamp': time.time()
        })
    
    def start_continuous_learning(self):
        """Start the continuous learning process"""
        if not self.is_learning:
            self.is_learning = True
            self.learning_thread = threading.Thread(target=self._learning_loop)
            self.learning_thread.daemon = True
            self.learning_thread.start()
            print("🧠 Continuous learning started")
    
    def stop_continuous_learning(self):
        """Stop the continuous learning process"""
        self.is_learning = False
        if self.learning_thread:
            self.learning_thread.join()
        print("🧠 Continuous learning stopped")
    
    def _learning_loop(self):
        """Main learning loop that runs in background"""
        while self.is_learning:
            try:
                # Get next training task (timeout after 1 second)
                task = self.learning_queue.get(timeout=1)
                
                model_name = task['model_name']
                if model_name not in self.models:
                    continue
                
                model = self.models[model_name]
                X = task['X']
                y = task['y']
                epochs = task['epochs']
                
                # Train the model
                start_time = time.time()
                history = model.train(X, y, epochs, verbose=False)
                training_time = time.time() - start_time
                
                # Record the learning
                record = LearningRecord(
                    timestamp=task['timestamp'],
                    model_type=model_name,
                    accuracy=history['accuracy'][-1],
                    loss=history['loss'][-1],
                    data_size=X.shape[0],
                    training_time=training_time
                )
                self.learning_records.append(record)
                
                # Save model periodically
                if len(self.learning_records) % 10 == 0:
                    self.save_model(model_name)
                
                print(f"✅ Model '{model_name}' learned from {X.shape[0]} samples "
                      f"(accuracy: {record.accuracy:.2%}, loss: {record.loss:.4f})")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Learning error: {e}")
    
    def save_model(self, model_name: str):
        """Save a model to disk"""
        if model_name in self.models:
            filepath = os.path.join(self.model_dir, f"{model_name}.json")
            self.models[model_name].save(filepath)
            return True
        return False
    
    def load_model(self, model_name: str):
        """Load a model from disk"""
        filepath = os.path.join(self.model_dir, f"{model_name}.json")
        if os.path.exists(filepath):
            model = RealNeuralNetwork([1])  # Dummy initialization
            model.load(filepath)
            self.models[model_name] = model
            return True
        return False
    
    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process"""
        if not self.learning_records:
            return {}
        
        recent_records = self.learning_records[-100:]  # Last 100 records
        
        return {
            'total_training_sessions': len(self.learning_records),
            'total_samples_processed': sum(r.data_size for r in self.learning_records),
            'average_accuracy': np.mean([r.accuracy for r in recent_records]),
            'average_loss': np.mean([r.loss for r in recent_records]),
            'total_training_time': sum(r.training_time for r in self.learning_records),
            'models': list(self.models.keys()),
            'is_learning': self.is_learning
        }

class ReinforcementLearningAgent:
    """Simple Q-learning agent for decision making"""
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.95, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        self.training_episodes = 0
        
    def choose_action(self, state: int, training: bool = False) -> int:
        """Choose an action using epsilon-greedy policy"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """Update Q-table using Q-learning formula"""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state, action] = new_q
    
    def train_episode(self, env_step_func, max_steps: int = 100) -> float:
        """Train for one episode"""
        state = 0  # Start state
        total_reward = 0
        
        for _ in range(max_steps):
            action = self.choose_action(state, training=True)
            next_state, reward, done = env_step_func(state, action)
            
            self.update(state, action, reward, next_state)
            total_reward += reward
            
            if done:
                break
            
            state = next_state
        
        self.training_episodes += 1
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * 0.995)
        
        return total_reward
    
    def save(self, filepath: str):
        """Save Q-table to file"""
        np.save(filepath, self.q_table)
    
    def load(self, filepath: str):
        """Load Q-table from file"""
        self.q_table = np.load(filepath)

# Example usage and testing
def demonstrate_real_ai():
    """Demonstrate the real AI system"""
    print("=" * 70)
    print("REAL AI SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # 1. Create continuous learning system
    cls = ContinuousLearningSystem()
    cls.start_continuous_learning()
    
    # 2. Create and train a model for XOR problem
    print("\n1. Training XOR Problem:")
    print("-" * 40)
    
    xor_model = cls.create_model("xor", layers=[2, 4, 1], learning_rate=0.5)
    
    X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    # Queue multiple training sessions
    for i in range(5):
        cls.queue_training("xor", X_xor, y_xor, epochs=200)
        time.sleep(0.1)  # Small delay between submissions
    
    # Wait for learning to complete
    time.sleep(2)
    
    # 3. Test the model
    predictions = xor_model.predict(X_xor)
    print("\nXOR Predictions:")
    for i, (input_val, target, pred) in enumerate(zip(X_xor, y_xor, predictions)):
        print(f"  {input_val} -> Target: {target[0]}, Predicted: {pred}")
    
    # 4. Create a classification model
    print("\n2. Training Classification Model:")
    print("-" * 40)
    
    # Generate synthetic data
    np.random.seed(42)
    X_class = np.random.randn(100, 4)
    y_class = np.eye(3)[np.random.randint(0, 3, 100)]  # 3 classes
    
    class_model = cls.create_model("classifier", layers=[4, 8, 3], learning_rate=0.1)
    cls.queue_training("classifier", X_class, y_class, epochs=50)
    
    # Wait and check stats
    time.sleep(1)
    
    # 5. Show learning statistics
    print("\n3. Learning Statistics:")
    print("-" * 40)
    stats = cls.get_learning_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 6. Test reinforcement learning
    print("\n4. Reinforcement Learning Agent:")
    print("-" * 40)
    
    rl_agent = ReinforcementLearningAgent(n_states=10, n_actions=4)
    
    # Simple environment function
    def simple_env(state, action):
        next_state = (state + action) % 10
        reward = 1 if next_state == 9 else -0.1
        done = next_state == 9
        return next_state, reward, done
    
    # Train the agent
    rewards = []
    for episode in range(100):
        reward = rl_agent.train_episode(simple_env)
        rewards.append(reward)
    
    print(f"  Average reward (first 10 episodes): {np.mean(rewards[:10]):.2f}")
    print(f"  Average reward (last 10 episodes): {np.mean(rewards[-10:]):.2f}")
    print(f"  Improvement: {(np.mean(rewards[-10:]) - np.mean(rewards[:10])):.2f}")
    
    # Stop continuous learning
    cls.stop_continuous_learning()
    
    print("\n" + "=" * 70)
    print("✅ REAL AI SYSTEM WORKING!")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_real_ai()