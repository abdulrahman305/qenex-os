#!/usr/bin/env python3
"""
QENEX Self-Improving AI/ML System
Real machine learning implementation with fraud detection and risk assessment
"""

import asyncio
import json
import logging
import pickle
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================================================
# Fraud Detection System
# ============================================================================

@dataclass
class TransactionFeatures:
    """Features extracted from transaction for ML processing"""
    amount: float
    hour_of_day: int
    day_of_week: int
    merchant_risk_score: float
    user_avg_transaction: float
    user_transaction_count: int
    location_risk_score: float
    time_since_last_transaction: float
    unusual_amount: float  # Deviation from user's normal
    velocity_score: float  # Transaction frequency

class FraudDetectionModel:
    """Real-time fraud detection using ensemble methods"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_estimators=100
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.performance_metrics = {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'last_updated': None
        }
        self.transaction_history = deque(maxlen=10000)
        self.user_profiles = {}
        
    def extract_features(self, transaction: Dict[str, Any]) -> np.ndarray:
        """Extract features from transaction"""
        
        # Get user profile
        user_id = transaction.get('user_id', 'unknown')
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'avg_amount': 100.0,
                'transaction_count': 0,
                'last_transaction_time': 0,
                'typical_hours': [],
                'typical_merchants': set()
            }
        
        profile = self.user_profiles[user_id]
        
        # Extract time features
        timestamp = transaction.get('timestamp', time.time())
        dt = datetime.fromtimestamp(timestamp)
        hour_of_day = dt.hour
        day_of_week = dt.weekday()
        
        # Amount features
        amount = float(transaction.get('amount', 0))
        avg_amount = profile['avg_amount']
        unusual_amount = abs(amount - avg_amount) / (avg_amount + 1)
        
        # Velocity features
        time_since_last = timestamp - profile['last_transaction_time'] if profile['last_transaction_time'] else 3600
        velocity_score = 1 / (time_since_last + 1) * 1000
        
        # Risk scores
        merchant = transaction.get('merchant', 'unknown')
        merchant_risk = self._get_merchant_risk_score(merchant)
        location = transaction.get('location', 'unknown')
        location_risk = self._get_location_risk_score(location)
        
        features = TransactionFeatures(
            amount=amount,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            merchant_risk_score=merchant_risk,
            user_avg_transaction=avg_amount,
            user_transaction_count=profile['transaction_count'],
            location_risk_score=location_risk,
            time_since_last_transaction=time_since_last,
            unusual_amount=unusual_amount,
            velocity_score=velocity_score
        )
        
        # Update user profile
        profile['transaction_count'] += 1
        profile['avg_amount'] = (profile['avg_amount'] * (profile['transaction_count'] - 1) + amount) / profile['transaction_count']
        profile['last_transaction_time'] = timestamp
        profile['typical_hours'].append(hour_of_day)
        profile['typical_merchants'].add(merchant)
        
        # Store in history
        self.transaction_history.append(transaction)
        
        return np.array([
            features.amount,
            features.hour_of_day,
            features.day_of_week,
            features.merchant_risk_score,
            features.user_avg_transaction,
            features.user_transaction_count,
            features.location_risk_score,
            features.time_since_last_transaction,
            features.unusual_amount,
            features.velocity_score
        ]).reshape(1, -1)
        
    def _get_merchant_risk_score(self, merchant: str) -> float:
        """Calculate merchant risk score based on historical data"""
        # High-risk merchant categories
        high_risk = ['gambling', 'crypto', 'adult', 'pharmacy']
        medium_risk = ['travel', 'electronics', 'jewelry']
        
        merchant_lower = merchant.lower()
        
        if any(category in merchant_lower for category in high_risk):
            return 0.8
        elif any(category in merchant_lower for category in medium_risk):
            return 0.5
        else:
            return 0.2
            
    def _get_location_risk_score(self, location: str) -> float:
        """Calculate location risk score"""
        # Simplified - in production would use geo-IP database
        high_risk_countries = ['NG', 'PK', 'ID', 'CN', 'IN']
        
        if location in high_risk_countries:
            return 0.9
        else:
            return 0.3
            
    def train(self, training_data: List[Dict[str, Any]], labels: List[int] = None):
        """Train the fraud detection model"""
        
        if not training_data:
            # Generate synthetic training data
            training_data, labels = self._generate_synthetic_data()
            
        # Extract features
        X = []
        for transaction in training_data:
            features = self.extract_features(transaction)
            X.append(features.flatten())
            
        X = np.array(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest (unsupervised)
        self.isolation_forest.fit(X_scaled)
        
        # Train random forest if labels provided
        if labels is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, labels, test_size=0.2, random_state=42
            )
            
            self.random_forest.fit(X_train, y_train)
            
            # Calculate performance metrics
            y_pred = self.random_forest.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary'
            )
            
            self.performance_metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'last_updated': datetime.now()
            })
            
            # Calculate feature importance
            importances = self.random_forest.feature_importances_
            feature_names = [
                'amount', 'hour_of_day', 'day_of_week', 'merchant_risk',
                'user_avg', 'user_count', 'location_risk', 'time_since_last',
                'unusual_amount', 'velocity'
            ]
            
            self.feature_importance = dict(zip(feature_names, importances))
            
        self.is_trained = True
        logger.info(f"Fraud detection model trained. Metrics: {self.performance_metrics}")
        
    def _generate_synthetic_data(self, n_samples: int = 10000) -> Tuple[List[Dict], List[int]]:
        """Generate synthetic transaction data for training"""
        
        data = []
        labels = []
        
        for i in range(n_samples):
            # Normal transactions (95%)
            if np.random.random() < 0.95:
                transaction = {
                    'user_id': f'user_{np.random.randint(1, 100)}',
                    'amount': np.random.lognormal(3, 1.5),
                    'timestamp': time.time() - np.random.randint(0, 86400 * 30),
                    'merchant': np.random.choice(['grocery', 'gas', 'restaurant', 'retail']),
                    'location': np.random.choice(['US', 'UK', 'DE', 'FR'])
                }
                data.append(transaction)
                labels.append(0)
            else:
                # Fraudulent transactions (5%)
                transaction = {
                    'user_id': f'user_{np.random.randint(1, 100)}',
                    'amount': np.random.lognormal(5, 2),  # Higher amounts
                    'timestamp': time.time() - np.random.randint(0, 3600),  # Recent
                    'merchant': np.random.choice(['gambling', 'crypto', 'unknown']),
                    'location': np.random.choice(['NG', 'PK', 'ID'])
                }
                data.append(transaction)
                labels.append(1)
                
        return data, labels
        
    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Predict if transaction is fraudulent"""
        
        if not self.is_trained:
            self.train([])
            
        # Extract features
        features = self.extract_features(transaction)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from both models
        isolation_score = self.isolation_forest.decision_function(features_scaled)[0]
        isolation_pred = self.isolation_forest.predict(features_scaled)[0]
        
        # Random forest probability
        if hasattr(self.random_forest, 'predict_proba'):
            rf_proba = self.random_forest.predict_proba(features_scaled)[0][1]
        else:
            rf_proba = 0.5
            
        # Combine predictions
        is_anomaly = isolation_pred == -1
        fraud_probability = (rf_proba + (1 if is_anomaly else 0)) / 2
        
        # Determine risk level
        if fraud_probability > 0.8:
            risk_level = "HIGH"
            action = "BLOCK"
        elif fraud_probability > 0.5:
            risk_level = "MEDIUM"
            action = "REVIEW"
        else:
            risk_level = "LOW"
            action = "APPROVE"
            
        result = {
            'transaction_id': transaction.get('id', 'unknown'),
            'fraud_probability': float(fraud_probability),
            'risk_level': risk_level,
            'action': action,
            'isolation_score': float(isolation_score),
            'timestamp': datetime.now().isoformat()
        }
        
        # Self-improvement: Store prediction for future training
        self._store_prediction(transaction, result)
        
        return result
        
    def _store_prediction(self, transaction: Dict, prediction: Dict):
        """Store prediction for self-improvement"""
        # In production, would store in database for periodic retraining
        pass
        
    def self_improve(self):
        """Self-improvement through continuous learning"""
        
        if len(self.transaction_history) < 100:
            return
            
        # Extract recent transactions
        recent_transactions = list(self.transaction_history)[-1000:]
        
        # Retrain with recent data
        self.train(recent_transactions)
        
        logger.info("Model self-improved with recent transaction data")

# ============================================================================
# Risk Assessment System
# ============================================================================

class RiskAssessmentModel:
    """Credit risk and operational risk assessment"""
    
    def __init__(self):
        self.credit_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.operational_risk_factors = {}
        self.risk_history = deque(maxlen=1000)
        self.is_trained = False
        
    def assess_credit_risk(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess credit risk for a customer"""
        
        # Extract features
        features = self._extract_credit_features(customer_data)
        
        if not self.is_trained:
            self._train_credit_model()
            
        # Predict default probability
        risk_score = self.credit_model.predict(features.reshape(1, -1))[0]
        
        # Calculate credit limit
        income = customer_data.get('annual_income', 50000)
        credit_limit = self._calculate_credit_limit(risk_score, income)
        
        # Determine risk category
        if risk_score < 0.3:
            category = "LOW"
        elif risk_score < 0.6:
            category = "MEDIUM"
        else:
            category = "HIGH"
            
        result = {
            'customer_id': customer_data.get('id', 'unknown'),
            'risk_score': float(risk_score),
            'risk_category': category,
            'recommended_credit_limit': float(credit_limit),
            'interest_rate': self._calculate_interest_rate(risk_score),
            'timestamp': datetime.now().isoformat()
        }
        
        self.risk_history.append(result)
        return result
        
    def _extract_credit_features(self, customer_data: Dict) -> np.ndarray:
        """Extract features for credit risk assessment"""
        
        features = [
            customer_data.get('age', 30),
            customer_data.get('annual_income', 50000),
            customer_data.get('employment_years', 5),
            customer_data.get('credit_history_length', 5),
            customer_data.get('num_credit_accounts', 2),
            customer_data.get('credit_utilization', 0.3),
            customer_data.get('payment_history_score', 0.8),
            customer_data.get('debt_to_income', 0.3)
        ]
        
        return np.array(features)
        
    def _train_credit_model(self):
        """Train credit risk model with synthetic data"""
        
        # Generate synthetic training data
        n_samples = 5000
        X = []
        y = []
        
        for _ in range(n_samples):
            features = [
                np.random.randint(18, 70),  # age
                np.random.lognormal(10.5, 0.5),  # income
                np.random.randint(0, 30),  # employment years
                np.random.randint(0, 20),  # credit history
                np.random.randint(1, 10),  # num accounts
                np.random.beta(2, 5),  # credit utilization
                np.random.beta(5, 2),  # payment history
                np.random.beta(2, 5)  # debt to income
            ]
            
            # Calculate risk score based on features
            risk = (
                0.2 * (1 - features[1] / 200000) +  # income
                0.2 * features[5] +  # credit utilization
                0.3 * (1 - features[6]) +  # payment history
                0.3 * features[7]  # debt to income
            )
            
            X.append(features)
            y.append(min(1, max(0, risk)))
            
        self.credit_model.fit(X, y)
        self.is_trained = True
        
    def _calculate_credit_limit(self, risk_score: float, income: float) -> float:
        """Calculate recommended credit limit"""
        
        base_limit = income * 0.3  # 30% of annual income
        risk_multiplier = 1 - risk_score
        
        return base_limit * risk_multiplier
        
    def _calculate_interest_rate(self, risk_score: float) -> float:
        """Calculate interest rate based on risk"""
        
        base_rate = 0.05  # 5% base rate
        risk_premium = risk_score * 0.15  # up to 15% additional
        
        return base_rate + risk_premium
        
    def assess_operational_risk(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess operational risk"""
        
        risk_factors = {
            'system_availability': operation_data.get('uptime', 0.99),
            'transaction_failure_rate': operation_data.get('failure_rate', 0.01),
            'security_incidents': operation_data.get('incidents', 0),
            'compliance_violations': operation_data.get('violations', 0),
            'employee_turnover': operation_data.get('turnover', 0.1)
        }
        
        # Calculate overall operational risk
        risk_score = (
            (1 - risk_factors['system_availability']) * 0.3 +
            risk_factors['transaction_failure_rate'] * 0.3 +
            min(1, risk_factors['security_incidents'] / 10) * 0.2 +
            min(1, risk_factors['compliance_violations'] / 5) * 0.1 +
            risk_factors['employee_turnover'] * 0.1
        )
        
        return {
            'risk_score': float(risk_score),
            'risk_factors': risk_factors,
            'recommendations': self._get_operational_recommendations(risk_factors),
            'timestamp': datetime.now().isoformat()
        }
        
    def _get_operational_recommendations(self, risk_factors: Dict) -> List[str]:
        """Generate recommendations based on risk factors"""
        
        recommendations = []
        
        if risk_factors['system_availability'] < 0.99:
            recommendations.append("Improve system redundancy and failover mechanisms")
            
        if risk_factors['transaction_failure_rate'] > 0.02:
            recommendations.append("Investigate and fix transaction processing issues")
            
        if risk_factors['security_incidents'] > 0:
            recommendations.append("Enhance security monitoring and incident response")
            
        return recommendations

# ============================================================================
# Self-Improving AI Controller
# ============================================================================

class SelfImprovingAI:
    """Main AI system with self-improvement capabilities"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.fraud_detector = FraudDetectionModel()
        self.risk_assessor = RiskAssessmentModel()
        
        self.performance_history = []
        self.improvement_schedule = {
            'last_improvement': None,
            'improvement_frequency': timedelta(hours=24),
            'min_data_points': 100
        }
        
        # Load existing models if available
        self.load_models()
        
    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process transaction through AI system"""
        
        # Fraud detection
        fraud_result = self.fraud_detector.predict(transaction)
        
        # Risk assessment if needed
        if fraud_result['risk_level'] in ['MEDIUM', 'HIGH']:
            customer_data = self._get_customer_data(transaction.get('user_id'))
            risk_result = self.risk_assessor.assess_credit_risk(customer_data)
        else:
            risk_result = None
            
        result = {
            'transaction_id': transaction.get('id'),
            'fraud_assessment': fraud_result,
            'risk_assessment': risk_result,
            'final_decision': self._make_final_decision(fraud_result, risk_result),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store for self-improvement
        self.performance_history.append(result)
        
        # Check if self-improvement is needed
        self._check_self_improvement()
        
        return result
        
    def _get_customer_data(self, user_id: str) -> Dict[str, Any]:
        """Get customer data for risk assessment"""
        # In production, would fetch from database
        return {
            'id': user_id,
            'age': 35,
            'annual_income': 75000,
            'employment_years': 8,
            'credit_history_length': 10,
            'num_credit_accounts': 3,
            'credit_utilization': 0.25,
            'payment_history_score': 0.9,
            'debt_to_income': 0.2
        }
        
    def _make_final_decision(self, fraud_result: Dict, risk_result: Optional[Dict]) -> str:
        """Make final decision based on all assessments"""
        
        if fraud_result['action'] == 'BLOCK':
            return 'REJECT'
        elif fraud_result['action'] == 'REVIEW':
            if risk_result and risk_result['risk_category'] == 'HIGH':
                return 'REJECT'
            else:
                return 'MANUAL_REVIEW'
        else:
            return 'APPROVE'
            
    def _check_self_improvement(self):
        """Check if models should self-improve"""
        
        now = datetime.now()
        
        if self.improvement_schedule['last_improvement'] is None:
            self.improvement_schedule['last_improvement'] = now
            return
            
        time_since_improvement = now - self.improvement_schedule['last_improvement']
        
        if (time_since_improvement >= self.improvement_schedule['improvement_frequency'] and
            len(self.performance_history) >= self.improvement_schedule['min_data_points']):
            
            self.self_improve()
            self.improvement_schedule['last_improvement'] = now
            
    def self_improve(self):
        """Trigger self-improvement for all models"""
        
        logger.info("Starting self-improvement cycle...")
        
        # Improve fraud detection
        self.fraud_detector.self_improve()
        
        # Save updated models
        self.save_models()
        
        # Clear old performance history
        self.performance_history = self.performance_history[-1000:]
        
        logger.info("Self-improvement cycle completed")
        
    def save_models(self):
        """Save models to disk"""
        
        try:
            # Save fraud detector
            with open(self.model_dir / "fraud_detector.pkl", "wb") as f:
                pickle.dump(self.fraud_detector, f)
                
            # Save risk assessor
            with open(self.model_dir / "risk_assessor.pkl", "wb") as f:
                pickle.dump(self.risk_assessor, f)
                
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            
    def load_models(self):
        """Load models from disk"""
        
        try:
            # Load fraud detector
            fraud_path = self.model_dir / "fraud_detector.pkl"
            if fraud_path.exists():
                with open(fraud_path, "rb") as f:
                    self.fraud_detector = pickle.load(f)
                    
            # Load risk assessor
            risk_path = self.model_dir / "risk_assessor.pkl"
            if risk_path.exists():
                with open(risk_path, "rb") as f:
                    self.risk_assessor = pickle.load(f)
                    
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        return {
            'fraud_detection': self.fraud_detector.performance_metrics,
            'feature_importance': self.fraud_detector.feature_importance,
            'total_transactions_processed': len(self.performance_history),
            'last_improvement': self.improvement_schedule['last_improvement'].isoformat() if self.improvement_schedule['last_improvement'] else None
        }

# ============================================================================
# Testing
# ============================================================================

async def test_ai_system():
    """Test AI/ML system"""
    
    ai = SelfImprovingAI()
    
    # Generate test transactions
    test_transactions = [
        {
            'id': f'tx_{i}',
            'user_id': f'user_{i % 10}',
            'amount': np.random.lognormal(3, 1.5),
            'timestamp': time.time(),
            'merchant': np.random.choice(['grocery', 'gas', 'restaurant', 'gambling']),
            'location': np.random.choice(['US', 'UK', 'NG'])
        }
        for i in range(100)
    ]
    
    # Process transactions
    results = []
    for tx in test_transactions:
        result = ai.process_transaction(tx)
        results.append(result)
        
        # Print some results
        if len(results) % 20 == 0:
            print(f"\nProcessed {len(results)} transactions")
            print(f"Latest result: {result['final_decision']}")
            print(f"Fraud probability: {result['fraud_assessment']['fraud_probability']:.2f}")
            
    # Get performance metrics
    metrics = ai.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(json.dumps(metrics, indent=2, default=str))
    
    # Trigger self-improvement
    ai.self_improve()
    
    print("\nAI system test completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_ai_system())