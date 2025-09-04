#!/usr/bin/env python3
"""
Machine Learning Fraud Detection with Real-time Scoring
Production-ready fraud detection using ensemble methods and deep learning
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime, timedelta
import json
import hashlib
import pickle
import time
from collections import deque, defaultdict
from enum import Enum
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import warnings
warnings.filterwarnings('ignore')

# Neural network components
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
FRAUD_SCORE_THRESHOLD = 0.7
ANOMALY_SCORE_THRESHOLD = 0.8
PATTERN_DETECTION_WINDOW = 3600  # 1 hour
MAX_VELOCITY_THRESHOLD = 10  # Max transactions per minute
FEATURE_CACHE_SIZE = 100000
MODEL_UPDATE_INTERVAL = 3600  # Update models every hour
BATCH_INFERENCE_SIZE = 1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudType(Enum):
    """Types of fraud detected"""
    ACCOUNT_TAKEOVER = "account_takeover"
    CARD_NOT_PRESENT = "card_not_present"
    IDENTITY_THEFT = "identity_theft"
    MONEY_LAUNDERING = "money_laundering"
    SYNTHETIC_IDENTITY = "synthetic_identity"
    TRANSACTION_FRAUD = "transaction_fraud"
    COLLUSION = "collusion"
    INSIDER_THREAT = "insider_threat"


class RiskLevel(Enum):
    """Risk classification levels"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5
    CRITICAL = 6


@dataclass
class TransactionFeatures:
    """Feature vector for fraud detection"""
    # Transaction attributes
    amount: float
    currency: str
    merchant_category: str
    transaction_type: str
    channel: str  # ATM, POS, Online, Mobile
    
    # Temporal features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    is_holiday: bool
    time_since_last_transaction: float
    
    # Velocity features
    transactions_last_hour: int
    transactions_last_day: int
    amount_last_hour: float
    amount_last_day: float
    unique_merchants_last_day: int
    
    # Geographic features
    country_code: str
    is_domestic: bool
    distance_from_last_transaction: float
    unusual_location: bool
    
    # Account features
    account_age_days: int
    account_balance: float
    credit_utilization: float
    failed_attempts_last_day: int
    
    # Behavioral features
    deviation_from_average_amount: float
    deviation_from_typical_time: float
    new_merchant: bool
    new_payment_method: bool
    
    # Network features
    sender_risk_score: float
    receiver_risk_score: float
    network_risk_score: float
    
    # Device/Session features
    device_fingerprint: str
    ip_reputation: float
    session_duration: float
    multiple_cards_same_device: bool
    
    def to_vector(self) -> np.ndarray:
        """Convert features to numerical vector"""
        vector = [
            self.amount,
            hash(self.currency) % 1000,
            hash(self.merchant_category) % 1000,
            hash(self.transaction_type) % 100,
            hash(self.channel) % 10,
            self.hour_of_day,
            self.day_of_week,
            int(self.is_weekend),
            int(self.is_holiday),
            self.time_since_last_transaction,
            self.transactions_last_hour,
            self.transactions_last_day,
            self.amount_last_hour,
            self.amount_last_day,
            self.unique_merchants_last_day,
            hash(self.country_code) % 1000,
            int(self.is_domestic),
            self.distance_from_last_transaction,
            int(self.unusual_location),
            self.account_age_days,
            self.account_balance,
            self.credit_utilization,
            self.failed_attempts_last_day,
            self.deviation_from_average_amount,
            self.deviation_from_typical_time,
            int(self.new_merchant),
            int(self.new_payment_method),
            self.sender_risk_score,
            self.receiver_risk_score,
            self.network_risk_score,
            hash(self.device_fingerprint) % 10000,
            self.ip_reputation,
            self.session_duration,
            int(self.multiple_cards_same_device)
        ]
        return np.array(vector, dtype=np.float32)


@dataclass
class FraudAlert:
    """Fraud detection alert"""
    alert_id: str
    transaction_id: str
    timestamp: float
    fraud_score: float
    risk_level: RiskLevel
    fraud_types: List[FraudType]
    confidence: float
    features: TransactionFeatures
    explanation: Dict[str, Any]
    recommended_action: str
    requires_manual_review: bool


class DeepFraudNet(nn.Module):
    """Deep neural network for fraud detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(DeepFraudNet, self).__init__()
        
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
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Apply attention
        attention_weights = self.attention(x)
        x = x * attention_weights
        
        # Main network
        return self.model(x)
    
    def get_feature_importance(self, x):
        """Get feature importance scores"""
        with torch.no_grad():
            return self.attention(x).cpu().numpy()


class GraphAnomalyDetector:
    """Graph-based anomaly detection for network fraud"""
    
    def __init__(self):
        self.transaction_graph = defaultdict(lambda: defaultdict(list))
        self.entity_features = {}
        self.suspicious_patterns = []
        
    def add_transaction(self, sender: str, receiver: str, amount: float, timestamp: float):
        """Add transaction to graph"""
        self.transaction_graph[sender][receiver].append({
            'amount': amount,
            'timestamp': timestamp
        })
    
    def detect_money_laundering_patterns(self) -> List[Dict]:
        """Detect potential money laundering patterns"""
        patterns = []
        
        # Detect layering (rapid movement through accounts)
        for sender in self.transaction_graph:
            chain = self._trace_transaction_chain(sender, max_depth=5)
            if len(chain) > 3:
                velocity = self._calculate_chain_velocity(chain)
                if velocity > 0.8:  # High velocity
                    patterns.append({
                        'type': 'layering',
                        'entities': chain,
                        'risk_score': velocity
                    })
        
        # Detect smurfing (structuring)
        for sender, receivers in self.transaction_graph.items():
            transactions = []
            for receiver, trans_list in receivers.items():
                transactions.extend(trans_list)
            
            if self._detect_structuring(transactions):
                patterns.append({
                    'type': 'smurfing',
                    'entity': sender,
                    'risk_score': 0.9
                })
        
        # Detect circular transactions
        cycles = self._find_cycles()
        for cycle in cycles:
            patterns.append({
                'type': 'circular_flow',
                'entities': cycle,
                'risk_score': 0.85
            })
        
        return patterns
    
    def _trace_transaction_chain(self, start: str, max_depth: int) -> List[str]:
        """Trace transaction chain from starting entity"""
        chain = [start]
        current = start
        depth = 0
        
        while depth < max_depth and current in self.transaction_graph:
            receivers = list(self.transaction_graph[current].keys())
            if not receivers:
                break
            
            # Follow the path with most recent transaction
            next_entity = max(receivers, key=lambda r: 
                            self.transaction_graph[current][r][-1]['timestamp'])
            
            if next_entity in chain:  # Avoid cycles
                break
            
            chain.append(next_entity)
            current = next_entity
            depth += 1
        
        return chain
    
    def _calculate_chain_velocity(self, chain: List[str]) -> float:
        """Calculate transaction velocity through chain"""
        if len(chain) < 2:
            return 0
        
        total_time = 0
        for i in range(len(chain) - 1):
            sender = chain[i]
            receiver = chain[i + 1]
            
            if receiver in self.transaction_graph[sender]:
                transactions = self.transaction_graph[sender][receiver]
                if len(transactions) >= 2:
                    time_diff = transactions[-1]['timestamp'] - transactions[0]['timestamp']
                    total_time += time_diff
        
        if total_time == 0:
            return 1.0  # Maximum velocity
        
        return min(1.0, len(chain) / (total_time / 3600))  # Transactions per hour
    
    def _detect_structuring(self, transactions: List[Dict]) -> bool:
        """Detect transaction structuring"""
        if len(transactions) < 3:
            return False
        
        amounts = [t['amount'] for t in transactions]
        
        # Check for transactions just below reporting threshold
        threshold = 10000  # $10,000 reporting threshold
        near_threshold = sum(1 for a in amounts if 0.8 * threshold <= a < threshold)
        
        if near_threshold / len(amounts) > 0.5:
            return True
        
        # Check for multiple identical amounts
        from collections import Counter
        amount_counts = Counter(amounts)
        max_count = max(amount_counts.values())
        
        if max_count >= 3 and max_count / len(amounts) > 0.3:
            return True
        
        return False
    
    def _find_cycles(self, max_cycle_length: int = 5) -> List[List[str]]:
        """Find cycles in transaction graph"""
        cycles = []
        visited = set()
        
        def dfs(node: str, path: List[str]):
            if len(path) > max_cycle_length:
                return
            
            if node in path:
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) >= 3:
                    cycles.append(cycle)
                return
            
            path.append(node)
            
            for neighbor in self.transaction_graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, path.copy())
            
            visited.add(node)
        
        for node in self.transaction_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles


class BehavioralAnalyzer:
    """Analyze user behavioral patterns"""
    
    def __init__(self):
        self.user_profiles = {}
        self.baseline_behaviors = {}
        self.anomaly_history = defaultdict(deque)
        
    def update_profile(self, user_id: str, features: TransactionFeatures):
        """Update user behavioral profile"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'transaction_count': 0,
                'total_amount': 0,
                'average_amount': 0,
                'typical_hours': [],
                'typical_merchants': set(),
                'typical_countries': set(),
                'last_updated': time.time()
            }
        
        profile = self.user_profiles[user_id]
        
        profile['transaction_count'] += 1
        profile['total_amount'] += features.amount
        profile['average_amount'] = profile['total_amount'] / profile['transaction_count']
        profile['typical_hours'].append(features.hour_of_day)
        profile['typical_merchants'].add(features.merchant_category)
        profile['typical_countries'].add(features.country_code)
        profile['last_updated'] = time.time()
        
        # Maintain sliding window of behaviors
        self.anomaly_history[user_id].append({
            'timestamp': time.time(),
            'features': features
        })
        
        # Keep only recent history
        while len(self.anomaly_history[user_id]) > 1000:
            self.anomaly_history[user_id].popleft()
    
    def calculate_behavioral_deviation(self, user_id: str, 
                                      features: TransactionFeatures) -> float:
        """Calculate deviation from normal behavior"""
        if user_id not in self.user_profiles:
            return 0.5  # Neutral score for new users
        
        profile = self.user_profiles[user_id]
        deviations = []
        
        # Amount deviation
        if profile['average_amount'] > 0:
            amount_dev = abs(features.amount - profile['average_amount']) / profile['average_amount']
            deviations.append(min(1.0, amount_dev))
        
        # Time deviation
        typical_hours = profile['typical_hours'][-100:]  # Last 100 transactions
        if typical_hours:
            hour_frequencies = np.bincount(typical_hours, minlength=24)
            hour_prob = hour_frequencies / hour_frequencies.sum()
            time_dev = 1 - hour_prob[features.hour_of_day]
            deviations.append(time_dev)
        
        # Merchant deviation
        if features.merchant_category not in profile['typical_merchants']:
            deviations.append(0.8)
        
        # Location deviation
        if features.country_code not in profile['typical_countries']:
            deviations.append(0.9)
        
        # Velocity deviation
        recent_transactions = [
            h for h in self.anomaly_history[user_id]
            if time.time() - h['timestamp'] < 3600
        ]
        
        if len(recent_transactions) > profile['transaction_count'] / 100:
            deviations.append(0.7)
        
        return np.mean(deviations) if deviations else 0


class MLFraudDetector:
    """Main ML fraud detection system"""
    
    def __init__(self):
        # Models
        self.deep_model = DeepFraudNet(input_dim=34)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Analyzers
        self.graph_detector = GraphAnomalyDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()
        
        # Feature engineering
        self.feature_cache = {}
        self.feature_statistics = {}
        
        # Model management
        self.model_version = "1.0.0"
        self.last_training = None
        self.is_trained = False
        
        # Real-time scoring queue
        self.scoring_queue = asyncio.Queue()
        self.alert_queue = asyncio.Queue()
        
        # Metrics
        self.metrics = {
            'transactions_scored': 0,
            'frauds_detected': 0,
            'false_positives': 0,
            'true_positives': 0,
            'average_score_time_ms': 0
        }
    
    async def initialize(self):
        """Initialize fraud detection system"""
        logger.info("Initializing ML fraud detection system")
        
        # Load pre-trained models if available
        await self.load_models()
        
        # Start background workers
        asyncio.create_task(self._scoring_worker())
        asyncio.create_task(self._model_updater())
        asyncio.create_task(self._alert_processor())
        
        logger.info("ML fraud detection system initialized")
    
    async def load_models(self):
        """Load pre-trained models"""
        try:
            # In production, load from model registry
            # self.deep_model.load_state_dict(torch.load('deep_fraud_model.pth'))
            # self.isolation_forest = joblib.load('isolation_forest.pkl')
            # self.random_forest = joblib.load('random_forest.pkl')
            # self.scaler = joblib.load('scaler.pkl')
            
            # For demo, train with synthetic data
            await self.train_models()
            
        except Exception as e:
            logger.warning(f"Could not load models: {e}. Training new models...")
            await self.train_models()
    
    async def train_models(self):
        """Train fraud detection models"""
        logger.info("Training fraud detection models")
        
        # Generate synthetic training data
        X_train, y_train = self._generate_synthetic_data(10000)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train isolation forest (unsupervised)
        self.isolation_forest.fit(X_scaled)
        
        # Train random forest (supervised)
        self.random_forest.fit(X_scaled, y_train)
        
        # Train deep learning model
        await self._train_deep_model(X_scaled, y_train)
        
        self.is_trained = True
        self.last_training = time.time()
        
        logger.info("Model training completed")
    
    def _generate_synthetic_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic fraud data for training"""
        # Create feature matrix
        X = np.random.randn(n_samples, 34)
        
        # Create labels (10% fraud)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        
        # Make fraudulent transactions different
        fraud_indices = np.where(y == 1)[0]
        X[fraud_indices] += np.random.randn(len(fraud_indices), 34) * 2
        
        return X, y
    
    async def _train_deep_model(self, X: np.ndarray, y: np.ndarray):
        """Train deep learning model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup optimizer
        optimizer = optim.Adam(self.deep_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training loop
        self.deep_model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.deep_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        self.deep_model.eval()
    
    async def score_transaction(self, transaction_id: str, 
                               features: TransactionFeatures) -> FraudAlert:
        """Score transaction for fraud risk"""
        start_time = time.time()
        
        # Extract feature vector
        feature_vector = features.to_vector()
        
        # Get ensemble predictions
        fraud_scores = await self._get_ensemble_scores(feature_vector)
        
        # Calculate final fraud score
        final_score = np.mean([
            fraud_scores['deep_learning'] * 0.4,
            fraud_scores['isolation_forest'] * 0.2,
            fraud_scores['random_forest'] * 0.3,
            fraud_scores['behavioral'] * 0.1
        ])
        
        # Determine risk level
        risk_level = self._calculate_risk_level(final_score)
        
        # Identify fraud types
        fraud_types = self._identify_fraud_types(features, fraud_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(features, fraud_scores)
        
        # Create alert
        alert = FraudAlert(
            alert_id=f"ALERT_{transaction_id}",
            transaction_id=transaction_id,
            timestamp=time.time(),
            fraud_score=final_score,
            risk_level=risk_level,
            fraud_types=fraud_types,
            confidence=self._calculate_confidence(fraud_scores),
            features=features,
            explanation=explanation,
            recommended_action=self._recommend_action(risk_level, final_score),
            requires_manual_review=risk_level.value >= RiskLevel.HIGH.value
        )
        
        # Update metrics
        self.metrics['transactions_scored'] += 1
        score_time_ms = (time.time() - start_time) * 1000
        self._update_average_score_time(score_time_ms)
        
        # Queue alert if high risk
        if alert.risk_level.value >= RiskLevel.HIGH.value:
            await self.alert_queue.put(alert)
            self.metrics['frauds_detected'] += 1
        
        # Update graph and behavioral models
        self.graph_detector.add_transaction(
            features.sender_risk_score,
            features.receiver_risk_score,
            features.amount,
            time.time()
        )
        
        return alert
    
    async def _get_ensemble_scores(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Get scores from all models"""
        scores = {}
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Deep learning score
        with torch.no_grad():
            tensor_input = torch.FloatTensor(feature_scaled)
            deep_score = self.deep_model(tensor_input).item()
            scores['deep_learning'] = deep_score
        
        # Isolation forest score
        iso_score = self.isolation_forest.decision_function(feature_scaled)[0]
        # Convert to probability-like score
        scores['isolation_forest'] = 1 / (1 + np.exp(-iso_score))
        
        # Random forest score
        if hasattr(self.random_forest, 'predict_proba'):
            rf_score = self.random_forest.predict_proba(feature_scaled)[0][1]
        else:
            rf_score = self.random_forest.predict(feature_scaled)[0]
        scores['random_forest'] = rf_score
        
        # Behavioral score (placeholder)
        scores['behavioral'] = np.random.random() * 0.5  # Would use actual behavioral analysis
        
        # Graph anomaly score
        ml_patterns = self.graph_detector.detect_money_laundering_patterns()
        scores['graph_anomaly'] = len(ml_patterns) / 10 if ml_patterns else 0
        
        return scores
    
    def _calculate_risk_level(self, fraud_score: float) -> RiskLevel:
        """Calculate risk level from fraud score"""
        if fraud_score < 0.2:
            return RiskLevel.VERY_LOW
        elif fraud_score < 0.4:
            return RiskLevel.LOW
        elif fraud_score < 0.6:
            return RiskLevel.MEDIUM
        elif fraud_score < 0.75:
            return RiskLevel.HIGH
        elif fraud_score < 0.9:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _identify_fraud_types(self, features: TransactionFeatures, 
                             scores: Dict[str, float]) -> List[FraudType]:
        """Identify specific fraud types"""
        fraud_types = []
        
        # Transaction fraud indicators
        if features.deviation_from_average_amount > 5:
            fraud_types.append(FraudType.TRANSACTION_FRAUD)
        
        # Account takeover indicators
        if features.unusual_location and features.new_payment_method:
            fraud_types.append(FraudType.ACCOUNT_TAKEOVER)
        
        # Money laundering indicators
        if scores.get('graph_anomaly', 0) > 0.5:
            fraud_types.append(FraudType.MONEY_LAUNDERING)
        
        # Card not present fraud
        if features.channel == 'Online' and features.multiple_cards_same_device:
            fraud_types.append(FraudType.CARD_NOT_PRESENT)
        
        return fraud_types if fraud_types else [FraudType.TRANSACTION_FRAUD]
    
    def _generate_explanation(self, features: TransactionFeatures, 
                            scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate human-readable explanation"""
        explanation = {
            'risk_factors': [],
            'model_scores': scores,
            'key_indicators': []
        }
        
        # Identify top risk factors
        if features.unusual_location:
            explanation['risk_factors'].append("Transaction from unusual location")
        
        if features.deviation_from_average_amount > 3:
            explanation['risk_factors'].append(
                f"Amount {features.deviation_from_average_amount:.1f}x higher than average"
            )
        
        if features.new_merchant:
            explanation['risk_factors'].append("First transaction with this merchant")
        
        if features.velocity > 5:
            explanation['risk_factors'].append("High transaction velocity detected")
        
        # Feature importance from deep model
        with torch.no_grad():
            feature_tensor = torch.FloatTensor(features.to_vector().reshape(1, -1))
            importance = self.deep_model.get_feature_importance(feature_tensor)[0]
            
            top_features_idx = np.argsort(importance)[-5:]
            explanation['key_indicators'] = [f"Feature_{i}" for i in top_features_idx]
        
        return explanation
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """Calculate confidence in fraud detection"""
        # Calculate agreement between models
        score_values = list(scores.values())
        mean_score = np.mean(score_values)
        std_score = np.std(score_values)
        
        # High agreement = high confidence
        if std_score < 0.1:
            return 0.9
        elif std_score < 0.2:
            return 0.7
        else:
            return 0.5
    
    def _recommend_action(self, risk_level: RiskLevel, fraud_score: float) -> str:
        """Recommend action based on risk"""
        if risk_level == RiskLevel.CRITICAL:
            return "BLOCK_IMMEDIATELY"
        elif risk_level == RiskLevel.VERY_HIGH:
            return "BLOCK_AND_REVIEW"
        elif risk_level == RiskLevel.HIGH:
            return "REQUIRE_ADDITIONAL_VERIFICATION"
        elif risk_level == RiskLevel.MEDIUM:
            return "FLAG_FOR_REVIEW"
        else:
            return "APPROVE"
    
    def _update_average_score_time(self, new_time_ms: float):
        """Update average scoring time metric"""
        count = self.metrics['transactions_scored']
        current_avg = self.metrics['average_score_time_ms']
        
        self.metrics['average_score_time_ms'] = (
            (current_avg * (count - 1) + new_time_ms) / count
        )
    
    async def _scoring_worker(self):
        """Background worker for transaction scoring"""
        while True:
            try:
                # Process scoring queue
                if not self.scoring_queue.empty():
                    transaction = await self.scoring_queue.get()
                    await self.score_transaction(
                        transaction['id'],
                        transaction['features']
                    )
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Scoring worker error: {e}")
    
    async def _model_updater(self):
        """Background worker for model updates"""
        while True:
            try:
                await asyncio.sleep(MODEL_UPDATE_INTERVAL)
                
                # Check if retraining is needed
                if self.last_training and \
                   time.time() - self.last_training > MODEL_UPDATE_INTERVAL:
                    logger.info("Initiating model retraining")
                    await self.train_models()
                
            except Exception as e:
                logger.error(f"Model updater error: {e}")
    
    async def _alert_processor(self):
        """Process high-risk alerts"""
        while True:
            try:
                alert = await asyncio.wait_for(
                    self.alert_queue.get(),
                    timeout=1.0
                )
                
                # In production, would send to case management system
                logger.warning(
                    f"HIGH RISK ALERT: Transaction {alert.transaction_id} "
                    f"Score: {alert.fraud_score:.2f}, Action: {alert.recommended_action}"
                )
                
                # Store alert for audit
                # await self.store_alert(alert)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Alert processor error: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get fraud detection metrics"""
        precision = (
            self.metrics['true_positives'] / 
            (self.metrics['true_positives'] + self.metrics['false_positives'])
            if (self.metrics['true_positives'] + self.metrics['false_positives']) > 0
            else 0
        )
        
        return {
            **self.metrics,
            'precision': precision,
            'model_version': self.model_version,
            'last_training': self.last_training
        }


async def main():
    """Test ML fraud detection system"""
    detector = MLFraudDetector()
    await detector.initialize()
    
    # Test transactions
    test_transactions = [
        # Normal transaction
        TransactionFeatures(
            amount=50.0,
            currency="USD",
            merchant_category="grocery",
            transaction_type="purchase",
            channel="POS",
            hour_of_day=14,
            day_of_week=2,
            is_weekend=False,
            is_holiday=False,
            time_since_last_transaction=3600,
            transactions_last_hour=1,
            transactions_last_day=5,
            amount_last_hour=50,
            amount_last_day=250,
            unique_merchants_last_day=3,
            country_code="US",
            is_domestic=True,
            distance_from_last_transaction=5.0,
            unusual_location=False,
            account_age_days=365,
            account_balance=5000,
            credit_utilization=0.3,
            failed_attempts_last_day=0,
            deviation_from_average_amount=0.1,
            deviation_from_typical_time=0.1,
            new_merchant=False,
            new_payment_method=False,
            sender_risk_score=0.1,
            receiver_risk_score=0.1,
            network_risk_score=0.1,
            device_fingerprint="device123",
            ip_reputation=0.9,
            session_duration=300,
            multiple_cards_same_device=False
        ),
        
        # Suspicious transaction
        TransactionFeatures(
            amount=5000.0,
            currency="USD",
            merchant_category="jewelry",
            transaction_type="purchase",
            channel="Online",
            hour_of_day=3,
            day_of_week=0,
            is_weekend=False,
            is_holiday=False,
            time_since_last_transaction=60,
            transactions_last_hour=10,
            transactions_last_day=50,
            amount_last_hour=15000,
            amount_last_day=50000,
            unique_merchants_last_day=20,
            country_code="XX",
            is_domestic=False,
            distance_from_last_transaction=5000.0,
            unusual_location=True,
            account_age_days=5,
            account_balance=100,
            credit_utilization=0.95,
            failed_attempts_last_day=5,
            deviation_from_average_amount=10.0,
            deviation_from_typical_time=5.0,
            new_merchant=True,
            new_payment_method=True,
            sender_risk_score=0.8,
            receiver_risk_score=0.9,
            network_risk_score=0.85,
            device_fingerprint="suspicious_device",
            ip_reputation=0.2,
            session_duration=10,
            multiple_cards_same_device=True
        )
    ]
    
    # Score transactions
    for i, features in enumerate(test_transactions):
        alert = await detector.score_transaction(f"TX_{i}", features)
        
        print(f"\nTransaction {i}:")
        print(f"  Fraud Score: {alert.fraud_score:.2f}")
        print(f"  Risk Level: {alert.risk_level.name}")
        print(f"  Fraud Types: {[ft.value for ft in alert.fraud_types]}")
        print(f"  Confidence: {alert.confidence:.2f}")
        print(f"  Action: {alert.recommended_action}")
        print(f"  Risk Factors: {alert.explanation['risk_factors']}")
    
    # Print metrics
    print(f"\nMetrics: {detector.get_metrics()}")


if __name__ == "__main__":
    asyncio.run(main())