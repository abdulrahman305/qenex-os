#!/usr/bin/env python3
"""
Self-Improving AI System
Continuous learning and adaptation for banking operations
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import json
import pickle
import logging
from collections import deque
import hashlib

# ML Libraries
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Track model performance over time"""
    model_id: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_samples: int
    timestamp: datetime
    improvement_rate: float

class AdaptiveModel:
    """Base class for self-improving models"""
    
    def __init__(self, model_type: str, initial_model=None):
        self.model_type = model_type
        self.model = initial_model
        self.version = "1.0.0"
        self.performance_history: List[ModelPerformance] = []
        self.training_data = deque(maxlen=10000)
        self.feedback_data = deque(maxlen=1000)
        self.last_update = datetime.now(timezone.utc)
        self.update_frequency = timedelta(hours=1)
        
    async def predict(self, features: np.ndarray) -> Any:
        """Make prediction with current model"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(features)
    
    async def update(self, X: np.ndarray, y: np.ndarray, feedback: Optional[Dict] = None):
        """Update model with new data"""
        # Store new training data
        for i in range(len(X)):
            self.training_data.append((X[i], y[i]))
        
        if feedback:
            self.feedback_data.append(feedback)
        
        # Check if update is needed
        if datetime.now(timezone.utc) - self.last_update > self.update_frequency:
            await self._retrain()
    
    async def _retrain(self):
        """Retrain model with accumulated data"""
        if len(self.training_data) < 100:
            return
        
        X = np.array([x for x, _ in self.training_data])
        y = np.array([y for _, y in self.training_data])
        
        # Train new model
        new_model = self._create_new_model()
        new_model.fit(X, y)
        
        # Evaluate performance
        score = cross_val_score(new_model, X, y, cv=5).mean()
        
        # Compare with current model
        if self.model is None or score > self._get_current_score():
            self.model = new_model
            self.version = self._increment_version()
            self.last_update = datetime.now(timezone.utc)
            logger.info(f"Model updated to version {self.version} with score {score:.3f}")
    
    def _create_new_model(self):
        """Create new model instance"""
        raise NotImplementedError
    
    def _get_current_score(self) -> float:
        """Get current model performance score"""
        if not self.performance_history:
            return 0.0
        return self.performance_history[-1].accuracy
    
    def _increment_version(self) -> str:
        """Increment model version"""
        parts = self.version.split('.')
        parts[2] = str(int(parts[2]) + 1)
        return '.'.join(parts)

class SelfImprovingFraudDetector(AdaptiveModel):
    """Self-improving fraud detection model"""
    
    def __init__(self):
        super().__init__("fraud_detection")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500
        )
        self.feature_importance = {}
        self.pattern_memory = deque(maxlen=1000)
        
    def _create_new_model(self):
        """Create neural network for fraud detection"""
        return MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500
        )
    
    async def learn_from_feedback(self, transaction_id: str, actual_fraud: bool, 
                                 predicted_fraud: bool):
        """Learn from actual fraud outcomes"""
        feedback = {
            'transaction_id': transaction_id,
            'actual': actual_fraud,
            'predicted': predicted_fraud,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.feedback_data.append(feedback)
        
        # Update pattern memory
        if actual_fraud:
            self.pattern_memory.append(transaction_id)
        
        # Trigger retraining if significant errors
        error_rate = self._calculate_error_rate()
        if error_rate > 0.1:  # 10% error threshold
            logger.info(f"High error rate {error_rate:.2%}, triggering retraining")
            await self._retrain()
    
    def _calculate_error_rate(self) -> float:
        """Calculate recent prediction error rate"""
        if len(self.feedback_data) < 10:
            return 0.0
        
        recent = list(self.feedback_data)[-100:]
        errors = sum(1 for f in recent if f['actual'] != f['predicted'])
        return errors / len(recent)

class SelfImprovingRiskScorer(AdaptiveModel):
    """Self-improving risk scoring model"""
    
    def __init__(self):
        super().__init__("risk_scoring")
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
        
    def _create_new_model(self):
        """Create gradient boosting model for risk scoring"""
        return GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    async def calculate_risk_score(self, features: np.ndarray) -> Tuple[float, str]:
        """Calculate risk score and category"""
        score = await self.predict(features.reshape(1, -1))
        score = float(score[0])
        
        # Determine risk category
        if score < self.risk_thresholds['low']:
            category = 'low'
        elif score < self.risk_thresholds['medium']:
            category = 'medium'
        elif score < self.risk_thresholds['high']:
            category = 'high'
        else:
            category = 'critical'
        
        return score, category
    
    async def adapt_thresholds(self, outcomes: List[Dict]):
        """Adapt risk thresholds based on outcomes"""
        # Analyze outcomes to optimize thresholds
        df = pd.DataFrame(outcomes)
        
        for category in self.risk_thresholds:
            # Find optimal threshold for each category
            optimal = self._find_optimal_threshold(df, category)
            if optimal is not None:
                old_threshold = self.risk_thresholds[category]
                self.risk_thresholds[category] = optimal
                logger.info(f"Adapted {category} threshold: {old_threshold:.3f} -> {optimal:.3f}")
    
    def _find_optimal_threshold(self, df: pd.DataFrame, category: str) -> Optional[float]:
        """Find optimal threshold for risk category"""
        # Simplified optimization - in production would use ROC analysis
        if len(df) < 100:
            return None
        
        scores = df['risk_score'].values
        outcomes = df['bad_outcome'].values
        
        best_threshold = None
        best_score = -1
        
        for threshold in np.linspace(0, 1, 100):
            predictions = scores > threshold
            accuracy = np.mean(predictions == outcomes)
            
            if accuracy > best_score:
                best_score = accuracy
                best_threshold = threshold
        
        return best_threshold

class SelfImprovingOptimizer:
    """Self-improving system optimizer"""
    
    def __init__(self):
        self.models: Dict[str, AdaptiveModel] = {
            'fraud': SelfImprovingFraudDetector(),
            'risk': SelfImprovingRiskScorer(),
            'performance': self._create_performance_model()
        }
        
        self.system_metrics = {
            'total_predictions': 0,
            'total_updates': 0,
            'avg_accuracy': 0.0,
            'improvement_rate': 0.0
        }
        
        self.optimization_history = deque(maxlen=1000)
        
    def _create_performance_model(self) -> AdaptiveModel:
        """Create performance prediction model"""
        model = AdaptiveModel("performance")
        model.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        return model
    
    async def optimize_system(self):
        """Continuously optimize the entire system"""
        while True:
            try:
                # Collect system metrics
                metrics = await self._collect_metrics()
                
                # Identify optimization opportunities
                opportunities = self._identify_optimizations(metrics)
                
                # Apply optimizations
                for opt in opportunities:
                    await self._apply_optimization(opt)
                
                # Update system metrics
                self._update_system_metrics(metrics)
                
                # Sleep before next optimization cycle
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> Dict:
        """Collect current system metrics"""
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'models': {}
        }
        
        for name, model in self.models.items():
            metrics['models'][name] = {
                'version': model.version,
                'last_update': model.last_update.isoformat(),
                'training_samples': len(model.training_data),
                'performance': model._get_current_score()
            }
        
        return metrics
    
    def _identify_optimizations(self, metrics: Dict) -> List[Dict]:
        """Identify optimization opportunities"""
        optimizations = []
        
        for name, model_metrics in metrics['models'].items():
            # Check if model needs update
            if model_metrics['training_samples'] > 1000:
                last_update = datetime.fromisoformat(model_metrics['last_update'])
                if datetime.now(timezone.utc) - last_update > timedelta(hours=6):
                    optimizations.append({
                        'type': 'model_update',
                        'model': name,
                        'reason': 'Stale model with sufficient data'
                    })
            
            # Check if performance is declining
            if model_metrics['performance'] < 0.8:
                optimizations.append({
                    'type': 'performance_boost',
                    'model': name,
                    'reason': f"Low performance: {model_metrics['performance']:.2%}"
                })
        
        return optimizations
    
    async def _apply_optimization(self, optimization: Dict):
        """Apply optimization to system"""
        opt_type = optimization['type']
        model_name = optimization.get('model')
        
        if opt_type == 'model_update' and model_name in self.models:
            model = self.models[model_name]
            await model._retrain()
            logger.info(f"Retrained {model_name} model: {optimization['reason']}")
        
        elif opt_type == 'performance_boost' and model_name in self.models:
            # Implement performance boosting strategies
            model = self.models[model_name]
            model.update_frequency = timedelta(minutes=30)  # More frequent updates
            logger.info(f"Boosted {model_name} update frequency: {optimization['reason']}")
        
        self.optimization_history.append({
            'optimization': optimization,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
    
    def _update_system_metrics(self, metrics: Dict):
        """Update overall system metrics"""
        self.system_metrics['total_predictions'] += 1
        
        # Calculate average accuracy across all models
        accuracies = [m['performance'] for m in metrics['models'].values()]
        self.system_metrics['avg_accuracy'] = np.mean(accuracies) if accuracies else 0.0
        
        # Calculate improvement rate
        if len(self.optimization_history) >= 2:
            recent = list(self.optimization_history)[-10:]
            improvements = sum(1 for o in recent if 'boost' in o['optimization']['type'])
            self.system_metrics['improvement_rate'] = improvements / len(recent)

class AutoMLBanking:
    """Automated machine learning for banking"""
    
    def __init__(self):
        self.optimizer = SelfImprovingOptimizer()
        self.active = False
        
    async def start(self):
        """Start the self-improving AI system"""
        self.active = True
        
        # Start optimization loop
        asyncio.create_task(self.optimizer.optimize_system())
        
        # Start model monitoring
        asyncio.create_task(self._monitor_models())
        
        logger.info("Self-improving AI system started")
    
    async def _monitor_models(self):
        """Monitor and report on model performance"""
        while self.active:
            try:
                metrics = await self.optimizer._collect_metrics()
                
                logger.info(f"System Performance:")
                logger.info(f"  Average Accuracy: {self.optimizer.system_metrics['avg_accuracy']:.2%}")
                logger.info(f"  Improvement Rate: {self.optimizer.system_metrics['improvement_rate']:.2%}")
                
                for name, model_metrics in metrics['models'].items():
                    logger.info(f"  {name}: v{model_metrics['version']} - {model_metrics['performance']:.2%}")
                
                await asyncio.sleep(600)  # Report every 10 minutes
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def process_transaction(self, transaction: Dict) -> Dict:
        """Process transaction through AI system"""
        # Extract features
        features = self._extract_features(transaction)
        
        # Get fraud prediction
        fraud_model = self.optimizer.models['fraud']
        fraud_prediction = await fraud_model.predict(features.reshape(1, -1))
        
        # Get risk score
        risk_model = self.optimizer.models['risk']
        risk_score, risk_category = await risk_model.calculate_risk_score(features)
        
        # Update models with transaction
        await fraud_model.update(features.reshape(1, -1), np.array([0]))  # Assume not fraud initially
        await risk_model.update(features.reshape(1, -1), np.array([risk_score]))
        
        return {
            'transaction_id': transaction.get('id', 'unknown'),
            'fraud_prediction': bool(fraud_prediction[0]),
            'risk_score': risk_score,
            'risk_category': risk_category,
            'model_versions': {
                'fraud': fraud_model.version,
                'risk': risk_model.version
            }
        }
    
    def _extract_features(self, transaction: Dict) -> np.ndarray:
        """Extract features from transaction"""
        return np.array([
            transaction.get('amount', 0),
            hash(transaction.get('merchant', '')) % 1000,
            hash(transaction.get('location', '')) % 100,
            transaction.get('hour', 12),
            transaction.get('day_of_week', 1),
            transaction.get('velocity_1h', 0),
            transaction.get('velocity_24h', 0)
        ])

# Example usage
async def main():
    """Example self-improving AI system"""
    ai_system = AutoMLBanking()
    await ai_system.start()
    
    # Simulate transactions
    for i in range(10):
        transaction = {
            'id': f'tx_{i}',
            'amount': np.random.uniform(10, 1000),
            'merchant': f'merchant_{np.random.randint(1, 100)}',
            'location': f'city_{np.random.randint(1, 50)}',
            'hour': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'velocity_1h': np.random.randint(0, 10),
            'velocity_24h': np.random.randint(0, 50)
        }
        
        result = await ai_system.process_transaction(transaction)
        print(f"Transaction {i}: Fraud={result['fraud_prediction']}, Risk={result['risk_category']}")
        
        await asyncio.sleep(1)
    
    # Let the system optimize
    await asyncio.sleep(10)
    
    print(f"\nSystem Metrics:")
    print(f"  Avg Accuracy: {ai_system.optimizer.system_metrics['avg_accuracy']:.2%}")
    print(f"  Improvement Rate: {ai_system.optimizer.system_metrics['improvement_rate']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())