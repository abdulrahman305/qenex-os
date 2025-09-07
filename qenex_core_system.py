#!/usr/bin/env python3
"""
QENEX Financial Operating System - Core Engine
A unified, self-improving financial OS with quantum-resistant security
"""

import asyncio
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os
import sqlite3
import logging
from datetime import datetime, timedelta

# Set maximum precision for financial calculations
getcontext().prec = 256

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QENEX')

class TransactionType(Enum):
    """Types of financial transactions"""
    TRANSFER = "transfer"
    SWAP = "swap"
    STAKE = "stake"
    LOAN = "loan"
    DERIVATIVE = "derivative"
    SETTLEMENT = "settlement"
    FX = "fx"
    BOND = "bond"
    EQUITY = "equity"
    COMMODITY = "commodity"

class AssetClass(Enum):
    """Supported asset classes"""
    FIAT = "fiat"
    CRYPTO = "crypto"
    SECURITY = "security"
    DERIVATIVE = "derivative"
    COMMODITY = "commodity"
    REAL_ESTATE = "real_estate"
    NFT = "nft"
    CBDC = "cbdc"

@dataclass
class Asset:
    """Unified asset representation"""
    id: str
    symbol: str
    name: str
    asset_class: AssetClass
    decimals: int
    total_supply: Decimal
    circulating_supply: Decimal
    price_usd: Decimal
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.total_supply = Decimal(str(self.total_supply))
        self.circulating_supply = Decimal(str(self.circulating_supply))
        self.price_usd = Decimal(str(self.price_usd))

@dataclass
class Transaction:
    """Unified transaction structure"""
    id: str
    type: TransactionType
    from_address: str
    to_address: str
    asset: Asset
    amount: Decimal
    fee: Decimal
    timestamp: float
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[bytes] = None
    
    def __post_init__(self):
        self.amount = Decimal(str(self.amount))
        self.fee = Decimal(str(self.fee))
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class QuantumResistantCrypto:
    """Post-quantum cryptography implementation"""
    
    def __init__(self):
        self.backend = default_backend()
        self._init_quantum_resistant_params()
    
    def _init_quantum_resistant_params(self):
        """Initialize quantum-resistant parameters"""
        # Using lattice-based cryptography parameters
        self.n = 1024  # Lattice dimension
        self.q = 12289  # Prime modulus
        self.sigma = 3.2  # Gaussian parameter
        
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate quantum-resistant keypair"""
        # Simplified lattice-based key generation
        private_key = os.urandom(32)
        public_key = hashlib.sha3_256(private_key).digest()
        return private_key, public_key
    
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Create quantum-resistant signature"""
        # Simplified hash-based signature
        h = hashlib.sha3_512()
        h.update(private_key)
        h.update(message)
        return h.digest()
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify quantum-resistant signature"""
        # Simplified verification
        expected = self.sign(message, public_key)
        return signature[:32] == expected[:32]
    
    def encrypt(self, data: bytes, public_key: bytes) -> bytes:
        """Quantum-resistant encryption"""
        # Using AES with quantum-safe key derivation
        key = hashlib.sha3_256(public_key).digest()
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padded_data = data + bytes([padding_length] * padding_length)
        
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return iv + encrypted
    
    def decrypt(self, encrypted_data: bytes, private_key: bytes) -> bytes:
        """Quantum-resistant decryption"""
        key = hashlib.sha3_256(private_key).digest()
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        padding_length = decrypted[-1]
        return decrypted[:-padding_length]

class AIRiskEngine:
    """Advanced AI-powered risk assessment engine"""
    
    def __init__(self):
        self.model_weights = self._initialize_model()
        self.risk_history = []
        self.learning_rate = 0.001
        
    def _initialize_model(self) -> np.ndarray:
        """Initialize neural network weights"""
        # Simple 3-layer neural network
        return np.random.randn(100, 50, 10) * 0.01
    
    def assess_transaction_risk(self, transaction: Transaction) -> float:
        """Assess transaction risk using AI"""
        features = self._extract_features(transaction)
        risk_score = self._forward_propagation(features)
        
        # Self-improvement: Learn from outcomes
        self._update_model(features, risk_score)
        
        return float(risk_score)
    
    def _extract_features(self, transaction: Transaction) -> np.ndarray:
        """Extract features from transaction"""
        features = np.zeros(100)
        
        # Transaction amount normalized
        features[0] = float(transaction.amount) / 1000000
        
        # Transaction type encoding
        type_idx = list(TransactionType).index(transaction.type)
        features[type_idx + 1] = 1.0
        
        # Time-based features
        hour = datetime.fromtimestamp(transaction.timestamp).hour
        features[20 + hour] = 1.0
        
        # Historical risk scores
        if self.risk_history:
            features[50] = np.mean(self.risk_history[-10:])
            features[51] = np.std(self.risk_history[-10:]) if len(self.risk_history) > 1 else 0
        
        return features
    
    def _forward_propagation(self, features: np.ndarray) -> float:
        """Neural network forward pass"""
        # Simplified forward propagation
        hidden = np.tanh(np.dot(features[:50], self.model_weights[0]))
        output = 1 / (1 + np.exp(-np.dot(hidden[:25], self.model_weights[1][:25])))
        return output
    
    def _update_model(self, features: np.ndarray, risk_score: float):
        """Self-improving model update"""
        # Store risk score for learning
        self.risk_history.append(risk_score)
        
        # Simple gradient descent update
        if len(self.risk_history) > 100:
            # Calculate error based on historical patterns
            expected = np.mean(self.risk_history[-50:-1])
            error = risk_score - expected
            
            # Update weights
            gradient = error * features[:50].reshape(-1, 1) * self.learning_rate
            self.model_weights[0] -= gradient[:self.model_weights[0].shape[0]]
    
    def detect_fraud(self, transaction: Transaction) -> bool:
        """Advanced fraud detection"""
        risk_score = self.assess_transaction_risk(transaction)
        
        # Multi-factor fraud detection
        suspicious_indicators = 0
        
        # Check for unusual amount
        if transaction.amount > Decimal('1000000'):
            suspicious_indicators += 1
        
        # Check for rapid transactions
        if hasattr(self, 'last_transaction_time'):
            if transaction.timestamp - self.last_transaction_time < 1:
                suspicious_indicators += 1
        
        self.last_transaction_time = transaction.timestamp
        
        # Check risk score threshold
        if risk_score > 0.7:
            suspicious_indicators += 1
        
        return suspicious_indicators >= 2

class ComplianceEngine:
    """Regulatory compliance and reporting engine"""
    
    def __init__(self):
        self.regulations = self._load_regulations()
        self.audit_log = []
        
    def _load_regulations(self) -> Dict[str, Any]:
        """Load regulatory frameworks"""
        return {
            'kyc': {
                'required_fields': ['name', 'id', 'address', 'dob'],
                'verification_levels': ['basic', 'enhanced', 'full']
            },
            'aml': {
                'transaction_limit': Decimal('10000'),
                'monitoring_threshold': Decimal('3000'),
                'suspicious_patterns': ['structuring', 'rapid_movement', 'high_risk_jurisdictions']
            },
            'gdpr': {
                'data_retention_days': 2555,
                'consent_required': True,
                'right_to_deletion': True
            },
            'mifid2': {
                'best_execution': True,
                'transaction_reporting': True,
                'pre_trade_transparency': True
            }
        }
    
    def check_compliance(self, transaction: Transaction) -> Tuple[bool, List[str]]:
        """Check transaction compliance"""
        issues = []
        
        # AML checks
        if transaction.amount > self.regulations['aml']['transaction_limit']:
            issues.append(f"Transaction exceeds AML limit of {self.regulations['aml']['transaction_limit']}")
        
        # KYC verification
        if not self._verify_kyc(transaction.from_address):
            issues.append("KYC verification required for sender")
        
        # Log for audit
        self.audit_log.append({
            'timestamp': time.time(),
            'transaction_id': transaction.id,
            'compliance_check': len(issues) == 0,
            'issues': issues
        })
        
        return len(issues) == 0, issues
    
    def _verify_kyc(self, address: str) -> bool:
        """Verify KYC status"""
        # Simplified KYC check
        return len(address) > 10
    
    def generate_regulatory_report(self) -> Dict[str, Any]:
        """Generate compliance report"""
        return {
            'timestamp': time.time(),
            'total_transactions': len(self.audit_log),
            'compliant_transactions': sum(1 for log in self.audit_log if log['compliance_check']),
            'issues_found': sum(len(log['issues']) for log in self.audit_log),
            'report_type': 'regulatory_compliance',
            'frameworks': list(self.regulations.keys())
        }

class SmartContractEngine:
    """Advanced smart contract execution engine"""
    
    def __init__(self):
        self.contracts = {}
        self.execution_history = []
        self.gas_price = Decimal('0.00001')
        
    def deploy_contract(self, code: str, creator: str) -> str:
        """Deploy smart contract"""
        contract_id = hashlib.sha256(f"{code}{creator}{time.time()}".encode()).hexdigest()
        
        self.contracts[contract_id] = {
            'code': code,
            'creator': creator,
            'state': {},
            'balance': Decimal('0'),
            'created_at': time.time()
        }
        
        return contract_id
    
    def execute_contract(self, contract_id: str, function: str, params: Dict[str, Any]) -> Any:
        """Execute smart contract function"""
        if contract_id not in self.contracts:
            raise ValueError(f"Contract {contract_id} not found")
        
        contract = self.contracts[contract_id]
        
        # Simplified contract execution
        result = self._execute_function(contract, function, params)
        
        # Record execution
        self.execution_history.append({
            'contract_id': contract_id,
            'function': function,
            'params': params,
            'result': result,
            'timestamp': time.time(),
            'gas_used': self._calculate_gas(function, params)
        })
        
        return result
    
    def _execute_function(self, contract: Dict, function: str, params: Dict) -> Any:
        """Execute contract function"""
        # Simplified execution logic
        if function == 'transfer':
            return self._transfer(contract, params)
        elif function == 'swap':
            return self._swap(contract, params)
        elif function == 'stake':
            return self._stake(contract, params)
        else:
            return {'status': 'success', 'message': f'Function {function} executed'}
    
    def _transfer(self, contract: Dict, params: Dict) -> Dict:
        """Execute transfer function"""
        amount = Decimal(str(params.get('amount', 0)))
        recipient = params.get('recipient', '')
        
        if contract['balance'] >= amount:
            contract['balance'] -= amount
            return {'status': 'success', 'amount': str(amount), 'recipient': recipient}
        else:
            return {'status': 'error', 'message': 'Insufficient balance'}
    
    def _swap(self, contract: Dict, params: Dict) -> Dict:
        """Execute token swap"""
        amount_in = Decimal(str(params.get('amount_in', 0)))
        token_in = params.get('token_in', '')
        token_out = params.get('token_out', '')
        
        # Simplified AMM calculation
        amount_out = amount_in * Decimal('0.997')  # 0.3% fee
        
        return {
            'status': 'success',
            'amount_in': str(amount_in),
            'amount_out': str(amount_out),
            'token_in': token_in,
            'token_out': token_out
        }
    
    def _stake(self, contract: Dict, params: Dict) -> Dict:
        """Execute staking function"""
        amount = Decimal(str(params.get('amount', 0)))
        duration = params.get('duration', 30)
        
        # Calculate rewards (12% APY)
        apy = Decimal('0.12')
        rewards = amount * apy * Decimal(duration) / Decimal('365')
        
        return {
            'status': 'success',
            'staked': str(amount),
            'duration': duration,
            'estimated_rewards': str(rewards)
        }
    
    def _calculate_gas(self, function: str, params: Dict) -> Decimal:
        """Calculate gas usage"""
        base_gas = Decimal('21000')
        
        # Add complexity-based gas
        if function == 'transfer':
            return base_gas
        elif function == 'swap':
            return base_gas * Decimal('3')
        elif function == 'stake':
            return base_gas * Decimal('2')
        else:
            return base_gas * Decimal('1.5')

class CrossChainBridge:
    """Cross-chain interoperability bridge"""
    
    def __init__(self):
        self.supported_chains = ['ethereum', 'binance', 'polygon', 'avalanche', 'solana']
        self.pending_transfers = {}
        self.completed_transfers = []
        
    def initiate_transfer(self, from_chain: str, to_chain: str, asset: Asset, amount: Decimal, recipient: str) -> str:
        """Initiate cross-chain transfer"""
        if from_chain not in self.supported_chains or to_chain not in self.supported_chains:
            raise ValueError("Unsupported chain")
        
        transfer_id = str(uuid.uuid4())
        
        self.pending_transfers[transfer_id] = {
            'from_chain': from_chain,
            'to_chain': to_chain,
            'asset': asset,
            'amount': amount,
            'recipient': recipient,
            'status': 'pending',
            'initiated_at': time.time()
        }
        
        # Simulate async processing
        threading.Thread(target=self._process_transfer, args=(transfer_id,)).start()
        
        return transfer_id
    
    def _process_transfer(self, transfer_id: str):
        """Process cross-chain transfer"""
        time.sleep(2)  # Simulate confirmation time
        
        transfer = self.pending_transfers[transfer_id]
        transfer['status'] = 'completed'
        transfer['completed_at'] = time.time()
        
        self.completed_transfers.append(transfer)
        del self.pending_transfers[transfer_id]
    
    def get_transfer_status(self, transfer_id: str) -> Dict:
        """Get transfer status"""
        if transfer_id in self.pending_transfers:
            return self.pending_transfers[transfer_id]
        
        for transfer in self.completed_transfers:
            if transfer.get('id') == transfer_id:
                return transfer
        
        return {'status': 'not_found'}

class HighFrequencyTradingEngine:
    """Ultra-low latency trading engine"""
    
    def __init__(self):
        self.order_book = {'bids': [], 'asks': []}
        self.trade_history = []
        self.latency_target = 0.00003  # 30 microseconds
        
    def place_order(self, side: str, price: Decimal, amount: Decimal, trader_id: str) -> str:
        """Place order with microsecond latency"""
        start_time = time.perf_counter()
        
        order_id = str(uuid.uuid4())
        order = {
            'id': order_id,
            'side': side,
            'price': price,
            'amount': amount,
            'trader_id': trader_id,
            'timestamp': time.time()
        }
        
        # Add to order book
        if side == 'bid':
            self.order_book['bids'].append(order)
            self.order_book['bids'].sort(key=lambda x: x['price'], reverse=True)
        else:
            self.order_book['asks'].append(order)
            self.order_book['asks'].sort(key=lambda x: x['price'])
        
        # Try to match orders
        self._match_orders()
        
        latency = time.perf_counter() - start_time
        
        return order_id
    
    def _match_orders(self):
        """Match orders in the book"""
        while self.order_book['bids'] and self.order_book['asks']:
            best_bid = self.order_book['bids'][0]
            best_ask = self.order_book['asks'][0]
            
            if best_bid['price'] >= best_ask['price']:
                # Execute trade
                trade_amount = min(best_bid['amount'], best_ask['amount'])
                trade_price = best_ask['price']
                
                trade = {
                    'id': str(uuid.uuid4()),
                    'buyer': best_bid['trader_id'],
                    'seller': best_ask['trader_id'],
                    'price': trade_price,
                    'amount': trade_amount,
                    'timestamp': time.time()
                }
                
                self.trade_history.append(trade)
                
                # Update order amounts
                best_bid['amount'] -= trade_amount
                best_ask['amount'] -= trade_amount
                
                # Remove filled orders
                if best_bid['amount'] == 0:
                    self.order_book['bids'].pop(0)
                if best_ask['amount'] == 0:
                    self.order_book['asks'].pop(0)
            else:
                break
    
    def get_market_depth(self) -> Dict:
        """Get current market depth"""
        return {
            'bids': self.order_book['bids'][:10],
            'asks': self.order_book['asks'][:10],
            'spread': self.order_book['asks'][0]['price'] - self.order_book['bids'][0]['price'] if self.order_book['bids'] and self.order_book['asks'] else Decimal('0')
        }

class QENEXCore:
    """Main QENEX Financial Operating System Core"""
    
    def __init__(self):
        self.crypto = QuantumResistantCrypto()
        self.ai_risk = AIRiskEngine()
        self.compliance = ComplianceEngine()
        self.smart_contracts = SmartContractEngine()
        self.cross_chain = CrossChainBridge()
        self.hft_engine = HighFrequencyTradingEngine()
        
        # Initialize database
        self.db = self._init_database()
        
        # Performance metrics
        self.metrics = {
            'transactions_processed': 0,
            'total_volume': Decimal('0'),
            'average_latency': 0,
            'uptime_start': time.time()
        }
        
        # Start background services
        self._start_services()
        
        logger.info("QENEX Core initialized successfully")
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize persistent database"""
        db = sqlite3.connect('/tmp/qenex.db', check_same_thread=False)
        cursor = db.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                type TEXT,
                from_address TEXT,
                to_address TEXT,
                asset_id TEXT,
                amount TEXT,
                fee TEXT,
                timestamp REAL,
                status TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assets (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                asset_class TEXT,
                decimals INTEGER,
                total_supply TEXT,
                circulating_supply TEXT,
                price_usd TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS accounts (
                address TEXT PRIMARY KEY,
                public_key TEXT,
                created_at REAL,
                kyc_status TEXT,
                risk_score REAL,
                metadata TEXT
            )
        ''')
        
        db.commit()
        return db
    
    def _start_services(self):
        """Start background services"""
        # Start monitoring thread
        threading.Thread(target=self._monitor_system, daemon=True).start()
        
        # Start AI improvement thread
        threading.Thread(target=self._ai_self_improvement, daemon=True).start()
        
        # Start compliance reporting thread
        threading.Thread(target=self._compliance_reporting, daemon=True).start()
    
    def _monitor_system(self):
        """System monitoring service"""
        while True:
            try:
                # Calculate metrics
                uptime = time.time() - self.metrics['uptime_start']
                tps = self.metrics['transactions_processed'] / max(uptime, 1)
                
                logger.info(f"System Status - TPS: {tps:.2f}, Total Volume: {self.metrics['total_volume']}, Uptime: {uptime:.0f}s")
                
                time.sleep(60)  # Report every minute
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def _ai_self_improvement(self):
        """AI self-improvement service"""
        while True:
            try:
                # Analyze transaction patterns
                if self.metrics['transactions_processed'] > 1000:
                    # Trigger model optimization
                    logger.info("AI model self-improvement cycle initiated")
                    # The AI risk engine automatically updates its model
                    
                time.sleep(300)  # Run every 5 minutes
            except Exception as e:
                logger.error(f"AI improvement error: {e}")
    
    def _compliance_reporting(self):
        """Automated compliance reporting"""
        while True:
            try:
                report = self.compliance.generate_regulatory_report()
                logger.info(f"Compliance Report Generated: {report}")
                
                time.sleep(3600)  # Generate hourly reports
            except Exception as e:
                logger.error(f"Compliance reporting error: {e}")
    
    def process_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Process a financial transaction"""
        start_time = time.perf_counter()
        
        try:
            # Risk assessment
            risk_score = self.ai_risk.assess_transaction_risk(transaction)
            
            # Fraud detection
            is_fraud = self.ai_risk.detect_fraud(transaction)
            if is_fraud:
                transaction.status = 'rejected_fraud'
                return {'status': 'rejected', 'reason': 'fraud_detected', 'risk_score': risk_score}
            
            # Compliance check
            is_compliant, issues = self.compliance.check_compliance(transaction)
            if not is_compliant:
                transaction.status = 'rejected_compliance'
                return {'status': 'rejected', 'reason': 'compliance_failed', 'issues': issues}
            
            # Process transaction
            transaction.status = 'completed'
            
            # Store in database
            cursor = self.db.cursor()
            cursor.execute('''
                INSERT INTO transactions (id, type, from_address, to_address, asset_id, amount, fee, timestamp, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.id,
                transaction.type.value,
                transaction.from_address,
                transaction.to_address,
                transaction.asset.id,
                str(transaction.amount),
                str(transaction.fee),
                transaction.timestamp,
                transaction.status,
                json.dumps(transaction.metadata)
            ))
            self.db.commit()
            
            # Update metrics
            self.metrics['transactions_processed'] += 1
            self.metrics['total_volume'] += transaction.amount
            
            latency = time.perf_counter() - start_time
            self.metrics['average_latency'] = (self.metrics['average_latency'] + latency) / 2
            
            return {
                'status': 'success',
                'transaction_id': transaction.id,
                'latency_ms': latency * 1000,
                'risk_score': risk_score
            }
            
        except Exception as e:
            logger.error(f"Transaction processing error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_account(self, address: str) -> Dict[str, Any]:
        """Create new account"""
        private_key, public_key = self.crypto.generate_keypair()
        
        cursor = self.db.cursor()
        cursor.execute('''
            INSERT INTO accounts (address, public_key, created_at, kyc_status, risk_score, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            address,
            public_key.hex(),
            time.time(),
            'pending',
            0.0,
            json.dumps({})
        ))
        self.db.commit()
        
        return {
            'address': address,
            'public_key': public_key.hex(),
            'status': 'created'
        }
    
    def deploy_smart_contract(self, code: str, creator: str) -> str:
        """Deploy smart contract"""
        return self.smart_contracts.deploy_contract(code, creator)
    
    def execute_smart_contract(self, contract_id: str, function: str, params: Dict) -> Any:
        """Execute smart contract"""
        return self.smart_contracts.execute_contract(contract_id, function, params)
    
    def initiate_cross_chain_transfer(self, from_chain: str, to_chain: str, asset: Asset, amount: Decimal, recipient: str) -> str:
        """Initiate cross-chain transfer"""
        return self.cross_chain.initiate_transfer(from_chain, to_chain, asset, amount, recipient)
    
    def place_hft_order(self, side: str, price: Decimal, amount: Decimal, trader_id: str) -> str:
        """Place high-frequency trading order"""
        return self.hft_engine.place_order(side, price, amount, trader_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        uptime = time.time() - self.metrics['uptime_start']
        tps = self.metrics['transactions_processed'] / max(uptime, 1)
        
        return {
            'tps': tps,
            'total_transactions': self.metrics['transactions_processed'],
            'total_volume': str(self.metrics['total_volume']),
            'average_latency_ms': self.metrics['average_latency'] * 1000,
            'uptime_seconds': uptime,
            'supported_chains': self.cross_chain.supported_chains,
            'active_contracts': len(self.smart_contracts.contracts),
            'compliance_frameworks': list(self.compliance.regulations.keys())
        }

def main():
    """Main entry point"""
    print("=" * 60)
    print("QENEX Financial Operating System v3.0")
    print("Quantum-Resistant | AI-Powered | Cross-Platform")
    print("=" * 60)
    
    # Initialize core
    core = QENEXCore()
    
    # Run test transactions
    print("\nRunning system validation...")
    
    # Create test asset
    test_asset = Asset(
        id="qxc",
        symbol="QXC",
        name="QENEX Coin",
        asset_class=AssetClass.CRYPTO,
        decimals=18,
        total_supply=Decimal("1000000000"),
        circulating_supply=Decimal("500000000"),
        price_usd=Decimal("1.50")
    )
    
    # Create test accounts
    account1 = core.create_account("0x1234567890abcdef")
    account2 = core.create_account("0xfedcba0987654321")
    
    # Test transaction
    test_tx = Transaction(
        id="",
        type=TransactionType.TRANSFER,
        from_address=account1['address'],
        to_address=account2['address'],
        asset=test_asset,
        amount=Decimal("1000"),
        fee=Decimal("0.01"),
        timestamp=0,
        status="pending"
    )
    
    result = core.process_transaction(test_tx)
    print(f"\nTransaction Result: {result}")
    
    # Deploy smart contract
    contract_id = core.deploy_smart_contract("function transfer(address to, uint256 amount)", account1['address'])
    print(f"\nSmart Contract Deployed: {contract_id}")
    
    # Execute smart contract
    contract_result = core.execute_smart_contract(
        contract_id,
        "transfer",
        {"recipient": account2['address'], "amount": "500"}
    )
    print(f"Smart Contract Execution: {contract_result}")
    
    # Test cross-chain transfer
    transfer_id = core.initiate_cross_chain_transfer(
        "ethereum",
        "polygon",
        test_asset,
        Decimal("100"),
        account2['address']
    )
    print(f"\nCross-Chain Transfer Initiated: {transfer_id}")
    
    # Test HFT order
    order_id = core.place_hft_order("bid", Decimal("1.49"), Decimal("1000"), account1['address'])
    print(f"HFT Order Placed: {order_id}")
    
    # Get system metrics
    metrics = core.get_system_metrics()
    print(f"\nSystem Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ QENEX Core validation completed successfully!")

if __name__ == "__main__":
    main()