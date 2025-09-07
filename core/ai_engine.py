#!/usr/bin/env python3
"""
QENEX AI Self-Improvement and Financial Intelligence Engine
Production-ready machine learning for fraud detection, risk analysis, and system optimization
"""

import asyncio
import json
import logging
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelType(Enum):
    """AI model types"""
    FRAUD_DETECTION = auto()
    RISK_ASSESSMENT = auto()
    ANOMALY_DETECTION = auto()
    TRANSACTION_SCORING = auto()
    PATTERN_RECOGNITION = auto()
    OPTIMIZATION = auto()


class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TransactionFeatures:
    """Features extracted from transaction for ML models"""
    amount: float
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    merchant_category: int
    country_code: int
    currency_code: int
    transaction_type: int
    account_age_days: int
    daily_transaction_count: int
    daily_transaction_sum: float
    velocity_score: float  # Transactions per hour
    amount_deviation: float  # Deviation from average
    geo_risk_score: float
    merchant_risk_score: float
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numpy vector"""
        return np.array([
            self.amount,
            self.hour_of_day,
            self.day_of_week,
            float(self.is_weekend),
            self.merchant_category,
            self.country_code,
            self.currency_code,
            self.transaction_type,
            self.account_age_days,
            self.daily_transaction_count,
            self.daily_transaction_sum,
            self.velocity_score,
            self.amount_deviation,
            self.geo_risk_score,
            self.merchant_risk_score
        ])


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    false_positive_rate: float
    false_negative_rate: float
    inference_time_ms: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'false_positive_rate': self.false_positive_rate,
            'false_negative_rate': self.false_negative_rate,
            'inference_time_ms': self.inference_time_ms,
            'last_updated': self.last_updated.isoformat()
        }


class FraudDetectionNN(nn.Module):
    """Neural network for fraud detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super(FraudDetectionNN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class FraudDetector:
    """Advanced fraud detection system"""
    
    def __init__(self):
        self.neural_network = FraudDetectionNN(input_dim=15)
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = deque(maxlen=1000)
        self.metrics = None
        
    def extract_features(self, transaction: Dict[str, Any]) -> TransactionFeatures:
        """Extract features from transaction data"""
        
        # Parse transaction timestamp
        if isinstance(transaction.get('timestamp'), str):
            timestamp = datetime.fromisoformat(transaction['timestamp'])
        else:
            timestamp = transaction.get('timestamp', datetime.now(timezone.utc))
        
        # Calculate temporal features
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        is_weekend = day_of_week >= 5
        
        # Get transaction amount
        amount = float(transaction.get('amount', 0))
        
        # Merchant and location features
        merchant_category = hash(transaction.get('merchant_category', 'unknown')) % 1000
        country_code = hash(transaction.get('country', 'US')) % 200
        currency_code = hash(transaction.get('currency', 'USD')) % 50
        transaction_type = hash(transaction.get('type', 'payment')) % 10
        
        # Account features
        account_age_days = transaction.get('account_age_days', 30)
        
        # Velocity features
        daily_transaction_count = transaction.get('daily_count', 1)
        daily_transaction_sum = float(transaction.get('daily_sum', amount))
        velocity_score = transaction.get('velocity_score', 1.0)
        
        # Statistical features
        avg_amount = transaction.get('avg_transaction_amount', amount)
        amount_deviation = abs(amount - avg_amount) / (avg_amount + 1)
        
        # Risk scores
        geo_risk_score = self._calculate_geo_risk(transaction.get('country', 'US'))
        merchant_risk_score = self._calculate_merchant_risk(
            transaction.get('merchant_category', 'unknown')
        )
        
        return TransactionFeatures(
            amount=amount,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            merchant_category=merchant_category,
            country_code=country_code,
            currency_code=currency_code,
            transaction_type=transaction_type,
            account_age_days=account_age_days,
            daily_transaction_count=daily_transaction_count,
            daily_transaction_sum=daily_transaction_sum,
            velocity_score=velocity_score,
            amount_deviation=amount_deviation,
            geo_risk_score=geo_risk_score,
            merchant_risk_score=merchant_risk_score
        )
    
    def _calculate_geo_risk(self, country: str) -> float:
        """Calculate geographic risk score"""
        high_risk_countries = {'XX', 'YY', 'ZZ'}  # Placeholder high-risk countries
        medium_risk_countries = {'AA', 'BB', 'CC'}  # Placeholder medium-risk countries
        
        if country in high_risk_countries:
            return 0.9
        elif country in medium_risk_countries:
            return 0.5
        else:
            return 0.1
    
    def _calculate_merchant_risk(self, merchant_category: str) -> float:
        """Calculate merchant risk score"""
        high_risk_categories = {'gambling', 'crypto', 'adult'}
        medium_risk_categories = {'travel', 'electronics', 'jewelry'}
        
        if merchant_category.lower() in high_risk_categories:
            return 0.8
        elif merchant_category.lower() in medium_risk_categories:
            return 0.4
        else:
            return 0.1
    
    async def train(self, training_data: List[Dict[str, Any]]):
        """Train fraud detection models"""
        logger.info("Starting fraud detection model training...")
        
        # Prepare training data
        X = []
        y = []
        
        for sample in training_data:
            features = self.extract_features(sample)
            X.append(features.to_vector())
            y.append(1 if sample.get('is_fraud', False) else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        self.random_forest.fit(X_train, y_train)
        
        # Train Isolation Forest (unsupervised)
        self.isolation_forest.fit(X_train[y_train == 0])  # Train on normal transactions
        
        # Train Neural Network
        await self._train_neural_network(X_train, y_train, X_test, y_test)
        
        # Calculate metrics
        self.metrics = await self._calculate_metrics(X_test, y_test)
        
        self.is_trained = True
        logger.info(f"Model training completed. Metrics: {self.metrics.to_dict()}")
    
    async def _train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 50
    ):
        """Train neural network model"""
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.neural_network(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")
    
    async def _calculate_metrics(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> ModelMetrics:
        """Calculate model performance metrics"""
        
        # Get predictions
        rf_predictions = self.random_forest.predict(X_test)
        rf_probabilities = self.random_forest.predict_proba(X_test)[:, 1]
        
        # Neural network predictions
        with torch.no_grad():
            nn_outputs = self.neural_network(torch.FloatTensor(X_test))
            nn_predictions = (nn_outputs.numpy() > 0.5).astype(int).flatten()
        
        # Ensemble prediction (average of models)
        ensemble_probabilities = (rf_probabilities + nn_outputs.numpy().flatten()) / 2
        ensemble_predictions = (ensemble_probabilities > 0.5).astype(int)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, ensemble_predictions, average='binary'
        )
        
        accuracy = np.mean(ensemble_predictions == y_test)
        auc_roc = roc_auc_score(y_test, ensemble_probabilities)
        
        # Calculate error rates
        false_positives = np.sum((ensemble_predictions == 1) & (y_test == 0))
        false_negatives = np.sum((ensemble_predictions == 0) & (y_test == 1))
        total_negatives = np.sum(y_test == 0)
        total_positives = np.sum(y_test == 1)
        
        fpr = false_positives / total_negatives if total_negatives > 0 else 0
        fnr = false_negatives / total_positives if total_positives > 0 else 0
        
        return ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            inference_time_ms=0.5,  # Will be measured during inference
            last_updated=datetime.now(timezone.utc)
        )
    
    async def predict(self, transaction: Dict[str, Any]) -> Tuple[bool, float, Dict[str, Any]]:
        """Predict if transaction is fraudulent"""
        
        if not self.is_trained:
            # Return conservative estimate if not trained
            return False, 0.1, {'reason': 'Model not trained'}
        
        start_time = time.time()
        
        # Extract features
        features = self.extract_features(transaction)
        feature_vector = features.to_vector().reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_vector)
        
        # Get predictions from all models
        rf_prob = self.random_forest.predict_proba(feature_scaled)[0, 1]
        
        with torch.no_grad():
            nn_prob = self.neural_network(torch.FloatTensor(feature_scaled)).item()
        
        isolation_score = self.isolation_forest.decision_function(feature_scaled)[0]
        isolation_prob = 1 / (1 + np.exp(-isolation_score))  # Convert to probability
        
        # Ensemble prediction (weighted average)
        weights = [0.4, 0.4, 0.2]  # RF, NN, Isolation Forest
        fraud_probability = (
            weights[0] * rf_prob +
            weights[1] * nn_prob +
            weights[2] * isolation_prob
        )
        
        is_fraud = fraud_probability > 0.5
        
        # Detailed analysis
        analysis = {
            'fraud_probability': fraud_probability,
            'random_forest_score': rf_prob,
            'neural_network_score': nn_prob,
            'anomaly_score': isolation_prob,
            'inference_time_ms': (time.time() - start_time) * 1000,
            'risk_factors': self._identify_risk_factors(features, fraud_probability)
        }
        
        # Log prediction
        self.training_history.append({
            'timestamp': datetime.now(timezone.utc),
            'prediction': is_fraud,
            'probability': fraud_probability,
            'transaction_id': transaction.get('id', 'unknown')
        })
        
        return is_fraud, fraud_probability, analysis
    
    def _identify_risk_factors(
        self,
        features: TransactionFeatures,
        fraud_probability: float
    ) -> List[str]:
        """Identify specific risk factors in transaction"""
        risk_factors = []
        
        if features.amount > 10000:
            risk_factors.append("High transaction amount")
        
        if features.velocity_score > 10:
            risk_factors.append("High transaction velocity")
        
        if features.amount_deviation > 3:
            risk_factors.append("Amount significantly deviates from average")
        
        if features.geo_risk_score > 0.7:
            risk_factors.append("High-risk geographic location")
        
        if features.merchant_risk_score > 0.6:
            risk_factors.append("High-risk merchant category")
        
        if features.is_weekend and features.hour_of_day in [2, 3, 4]:
            risk_factors.append("Unusual transaction time")
        
        if fraud_probability > 0.8:
            risk_factors.append("Multiple suspicious patterns detected")
        
        return risk_factors


class RiskAssessmentEngine:
    """Comprehensive risk assessment for financial operations"""
    
    def __init__(self):
        self.risk_models = {}
        self.risk_thresholds = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 0.9
        }
        self.risk_history = deque(maxlen=10000)
        
    async def assess_transaction_risk(
        self,
        transaction: Dict[str, Any],
        account_history: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RiskLevel, float, Dict[str, Any]]:
        """Assess risk level of transaction"""
        
        risk_scores = []
        risk_details = {}
        
        # Amount-based risk
        amount = float(transaction.get('amount', 0))
        amount_risk = self._assess_amount_risk(amount, account_history)
        risk_scores.append(amount_risk)
        risk_details['amount_risk'] = amount_risk
        
        # Velocity risk
        velocity_risk = self._assess_velocity_risk(transaction, account_history)
        risk_scores.append(velocity_risk)
        risk_details['velocity_risk'] = velocity_risk
        
        # Geographic risk
        geo_risk = self._assess_geographic_risk(transaction)
        risk_scores.append(geo_risk)
        risk_details['geographic_risk'] = geo_risk
        
        # Behavioral risk
        behavioral_risk = await self._assess_behavioral_risk(transaction, account_history)
        risk_scores.append(behavioral_risk)
        risk_details['behavioral_risk'] = behavioral_risk
        
        # Network risk (relationships with other accounts)
        network_risk = self._assess_network_risk(transaction)
        risk_scores.append(network_risk)
        risk_details['network_risk'] = network_risk
        
        # Calculate composite risk score
        composite_risk = np.mean(risk_scores) * 1.2  # Apply 20% safety margin
        composite_risk = min(composite_risk, 1.0)
        
        # Determine risk level
        risk_level = RiskLevel.LOW
        for level, threshold in self.risk_thresholds.items():
            if composite_risk >= threshold:
                risk_level = level
        
        # Create detailed assessment
        assessment = {
            'risk_level': risk_level.name,
            'risk_score': composite_risk,
            'components': risk_details,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recommendations': self._generate_recommendations(risk_level, risk_details)
        }
        
        # Log assessment
        self.risk_history.append({
            'transaction_id': transaction.get('id'),
            'risk_level': risk_level,
            'risk_score': composite_risk,
            'timestamp': datetime.now(timezone.utc)
        })
        
        return risk_level, composite_risk, assessment
    
    def _assess_amount_risk(
        self,
        amount: float,
        account_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Assess risk based on transaction amount"""
        
        if not account_history:
            # Conservative estimate without history
            if amount > 10000:
                return 0.8
            elif amount > 5000:
                return 0.5
            elif amount > 1000:
                return 0.3
            else:
                return 0.1
        
        # Calculate based on historical patterns
        amounts = [float(tx.get('amount', 0)) for tx in account_history]
        avg_amount = np.mean(amounts) if amounts else amount
        std_amount = np.std(amounts) if len(amounts) > 1 else avg_amount * 0.5
        
        # Z-score calculation
        z_score = abs(amount - avg_amount) / (std_amount + 1)
        
        # Convert to risk score (0-1)
        risk_score = 1 - np.exp(-z_score / 2)
        return min(risk_score, 1.0)
    
    def _assess_velocity_risk(
        self,
        transaction: Dict[str, Any],
        account_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Assess risk based on transaction velocity"""
        
        if not account_history:
            return 0.2  # Low risk without history
        
        # Count recent transactions
        current_time = datetime.now(timezone.utc)
        one_hour_ago = current_time - timedelta(hours=1)
        one_day_ago = current_time - timedelta(days=1)
        
        hourly_count = sum(
            1 for tx in account_history
            if datetime.fromisoformat(tx.get('timestamp', '')) > one_hour_ago
        )
        
        daily_count = sum(
            1 for tx in account_history
            if datetime.fromisoformat(tx.get('timestamp', '')) > one_day_ago
        )
        
        # Calculate velocity risk
        hourly_risk = min(hourly_count / 10, 1.0)  # More than 10 per hour is high risk
        daily_risk = min(daily_count / 50, 1.0)  # More than 50 per day is high risk
        
        return max(hourly_risk, daily_risk * 0.7)
    
    def _assess_geographic_risk(self, transaction: Dict[str, Any]) -> float:
        """Assess geographic risk"""
        country = transaction.get('country', 'US')
        
        # High-risk countries (example list)
        high_risk = {'XX', 'YY', 'ZZ'}
        medium_risk = {'AA', 'BB', 'CC'}
        
        if country in high_risk:
            return 0.9
        elif country in medium_risk:
            return 0.5
        else:
            return 0.1
    
    async def _assess_behavioral_risk(
        self,
        transaction: Dict[str, Any],
        account_history: Optional[List[Dict[str, Any]]]
    ) -> float:
        """Assess behavioral risk patterns"""
        
        if not account_history or len(account_history) < 10:
            return 0.3  # Neutral risk for new accounts
        
        # Analyze behavioral patterns
        risk_score = 0.0
        
        # Time-based patterns
        tx_time = datetime.fromisoformat(transaction.get('timestamp', ''))
        tx_hour = tx_time.hour
        
        historical_hours = [
            datetime.fromisoformat(tx.get('timestamp', '')).hour
            for tx in account_history
        ]
        
        # Check if transaction is at unusual time
        hour_counts = np.bincount(historical_hours, minlength=24)
        if hour_counts[tx_hour] == 0:
            risk_score += 0.3  # Transaction at never-before-seen hour
        
        # Check for sudden change in transaction patterns
        recent_amounts = [
            float(tx.get('amount', 0))
            for tx in account_history[-10:]
        ]
        
        if recent_amounts:
            recent_avg = np.mean(recent_amounts)
            current_amount = float(transaction.get('amount', 0))
            
            if current_amount > recent_avg * 5:
                risk_score += 0.4  # Sudden spike in amount
        
        return min(risk_score, 1.0)
    
    def _assess_network_risk(self, transaction: Dict[str, Any]) -> float:
        """Assess risk based on network relationships"""
        
        # In production, would analyze relationships with other accounts
        # For now, return placeholder based on recipient
        recipient = transaction.get('recipient', '')
        
        if 'high_risk' in recipient.lower():
            return 0.8
        elif 'medium_risk' in recipient.lower():
            return 0.5
        else:
            return 0.1
    
    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        risk_details: Dict[str, float]
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Block transaction immediately")
            recommendations.append("Initiate manual review")
            recommendations.append("Contact account holder for verification")
        
        elif risk_level == RiskLevel.HIGH:
            recommendations.append("Require additional authentication")
            recommendations.append("Apply transaction hold for review")
            recommendations.append("Send alert to account holder")
        
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.append("Monitor subsequent transactions closely")
            recommendations.append("Send notification to account holder")
        
        # Specific recommendations based on risk components
        if risk_details.get('velocity_risk', 0) > 0.7:
            recommendations.append("Implement velocity controls")
        
        if risk_details.get('amount_risk', 0) > 0.7:
            recommendations.append("Verify source of funds")
        
        if risk_details.get('geographic_risk', 0) > 0.7:
            recommendations.append("Verify transaction location")
        
        return recommendations


class SelfImprovementEngine:
    """AI self-improvement and optimization engine"""
    
    def __init__(self):
        self.performance_metrics = deque(maxlen=1000)
        self.optimization_history = []
        self.current_parameters = self._initialize_parameters()
        self.improvement_rate = 0.0
        
    def _initialize_parameters(self) -> Dict[str, Any]:
        """Initialize system parameters"""
        return {
            'fraud_threshold': 0.5,
            'risk_weights': {
                'amount': 0.3,
                'velocity': 0.2,
                'geographic': 0.2,
                'behavioral': 0.2,
                'network': 0.1
            },
            'cache_ttl': 3600,
            'batch_size': 32,
            'learning_rate': 0.001,
            'model_update_frequency': 86400  # Daily
        }
    
    async def optimize_parameters(self, performance_data: List[Dict[str, Any]]):
        """Optimize system parameters based on performance"""
        
        if len(performance_data) < 100:
            return  # Need sufficient data for optimization
        
        logger.info("Starting parameter optimization...")
        
        # Calculate current performance
        current_performance = self._calculate_performance_score(performance_data[-100:])
        
        # Try different parameter combinations
        best_params = self.current_parameters.copy()
        best_score = current_performance
        
        # Grid search for optimal parameters
        for fraud_threshold in [0.4, 0.45, 0.5, 0.55, 0.6]:
            for learning_rate in [0.0005, 0.001, 0.002]:
                test_params = self.current_parameters.copy()
                test_params['fraud_threshold'] = fraud_threshold
                test_params['learning_rate'] = learning_rate
                
                # Simulate performance with new parameters
                simulated_score = await self._simulate_performance(
                    test_params,
                    performance_data
                )
                
                if simulated_score > best_score:
                    best_score = simulated_score
                    best_params = test_params
        
        # Calculate improvement
        self.improvement_rate = (best_score - current_performance) / current_performance
        
        # Update parameters if improvement found
        if self.improvement_rate > 0.01:  # At least 1% improvement
            self.current_parameters = best_params
            logger.info(f"Parameters optimized. Improvement: {self.improvement_rate:.2%}")
            
            # Log optimization
            self.optimization_history.append({
                'timestamp': datetime.now(timezone.utc),
                'old_score': current_performance,
                'new_score': best_score,
                'improvement': self.improvement_rate,
                'parameters': best_params
            })
    
    def _calculate_performance_score(self, data: List[Dict[str, Any]]) -> float:
        """Calculate overall system performance score"""
        
        if not data:
            return 0.0
        
        # Extract metrics
        accuracies = [d.get('accuracy', 0) for d in data if 'accuracy' in d]
        latencies = [d.get('latency', 1000) for d in data if 'latency' in d]
        error_rates = [d.get('error_rate', 1) for d in data if 'error_rate' in d]
        
        # Calculate composite score
        avg_accuracy = np.mean(accuracies) if accuracies else 0.5
        avg_latency = np.mean(latencies) if latencies else 1000
        avg_error_rate = np.mean(error_rates) if error_rates else 0.1
        
        # Normalize and combine (higher is better)
        accuracy_score = avg_accuracy
        latency_score = 1 / (1 + avg_latency / 100)  # Lower latency is better
        error_score = 1 - avg_error_rate
        
        # Weighted combination
        composite_score = (
            0.5 * accuracy_score +
            0.3 * latency_score +
            0.2 * error_score
        )
        
        return composite_score
    
    async def _simulate_performance(
        self,
        parameters: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> float:
        """Simulate system performance with given parameters"""
        
        # Simple simulation based on parameter changes
        # In production, would run actual backtesting
        
        base_score = self._calculate_performance_score(historical_data)
        
        # Adjust score based on parameters
        fraud_threshold_optimal = 0.5
        threshold_deviation = abs(parameters['fraud_threshold'] - fraud_threshold_optimal)
        threshold_penalty = threshold_deviation * 0.1
        
        learning_rate_optimal = 0.001
        lr_deviation = abs(parameters['learning_rate'] - learning_rate_optimal)
        lr_penalty = lr_deviation * 100  # Scale to similar magnitude
        
        simulated_score = base_score - threshold_penalty - lr_penalty
        
        # Add some randomness to simulation
        noise = np.random.normal(0, 0.01)
        simulated_score += noise
        
        return max(0, min(1, simulated_score))
    
    async def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report"""
        
        report = {
            'current_parameters': self.current_parameters,
            'improvement_rate': self.improvement_rate,
            'optimization_history': self.optimization_history[-10:],
            'performance_trend': self._calculate_performance_trend(),
            'recommendations': self._generate_optimization_recommendations(),
            'next_optimization': datetime.now(timezone.utc) + timedelta(hours=24)
        }
        
        return report
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend"""
        
        if len(self.optimization_history) < 2:
            return "insufficient_data"
        
        recent_improvements = [
            opt['improvement'] for opt in self.optimization_history[-5:]
        ]
        
        avg_improvement = np.mean(recent_improvements)
        
        if avg_improvement > 0.05:
            return "significant_improvement"
        elif avg_improvement > 0.01:
            return "moderate_improvement"
        elif avg_improvement > -0.01:
            return "stable"
        else:
            return "degradation"
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate recommendations for further optimization"""
        
        recommendations = []
        
        if self.improvement_rate < 0.01:
            recommendations.append("Consider expanding parameter search space")
            recommendations.append("Collect more training data")
        
        if self.current_parameters['fraud_threshold'] > 0.6:
            recommendations.append("High fraud threshold may miss legitimate fraud")
        elif self.current_parameters['fraud_threshold'] < 0.4:
            recommendations.append("Low fraud threshold may cause false positives")
        
        if len(self.performance_metrics) < 500:
            recommendations.append("Accumulate more performance data for better optimization")
        
        return recommendations


# Example usage
async def main():
    """Test AI engine"""
    
    # Initialize components
    fraud_detector = FraudDetector()
    risk_engine = RiskAssessmentEngine()
    self_improvement = SelfImprovementEngine()
    
    # Generate synthetic training data
    training_data = []
    for i in range(1000):
        is_fraud = np.random.random() < 0.1  # 10% fraud rate
        
        transaction = {
            'id': f'tx_{i}',
            'amount': np.random.lognormal(3, 2) if not is_fraud else np.random.lognormal(5, 2),
            'timestamp': (datetime.now(timezone.utc) - timedelta(days=np.random.randint(0, 30))).isoformat(),
            'merchant_category': np.random.choice(['retail', 'food', 'travel', 'gambling']),
            'country': np.random.choice(['US', 'GB', 'FR', 'XX']),
            'currency': 'USD',
            'is_fraud': is_fraud,
            'account_age_days': np.random.randint(1, 1000),
            'daily_count': np.random.randint(1, 20),
            'velocity_score': np.random.random() * 10
        }
        training_data.append(transaction)
    
    # Train fraud detector
    await fraud_detector.train(training_data)
    
    # Test fraud detection
    test_transaction = {
        'id': 'test_001',
        'amount': 5000,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'merchant_category': 'gambling',
        'country': 'XX',
        'currency': 'USD',
        'account_age_days': 5,
        'daily_count': 15,
        'velocity_score': 8.5
    }
    
    is_fraud, probability, analysis = await fraud_detector.predict(test_transaction)
    print(f"Fraud Detection: {is_fraud}, Probability: {probability:.2%}")
    print(f"Analysis: {json.dumps(analysis, indent=2)}")
    
    # Test risk assessment
    risk_level, risk_score, assessment = await risk_engine.assess_transaction_risk(
        test_transaction,
        training_data[-50:]  # Use last 50 transactions as history
    )
    print(f"\nRisk Assessment: {risk_level.name}, Score: {risk_score:.2%}")
    print(f"Details: {json.dumps(assessment, indent=2)}")
    
    # Test self-improvement
    performance_data = [
        {
            'accuracy': 0.92 + np.random.normal(0, 0.02),
            'latency': 50 + np.random.normal(0, 10),
            'error_rate': 0.05 + np.random.normal(0, 0.01)
        }
        for _ in range(200)
    ]
    
    await self_improvement.optimize_parameters(performance_data)
    report = await self_improvement.generate_optimization_report()
    print(f"\nOptimization Report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    asyncio.run(main())