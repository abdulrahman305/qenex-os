#!/usr/bin/env python3
"""
QENEX Unified Banking Operating System
Integration of all components into a complete banking platform
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

# Import QENEX components
from banking_protocols import BankingProtocolManager, ISO20022MessageType, SWIFTMessageType, SEPATransactionType
from ai_ml_system import SelfImprovingAI, FraudDetectionModel, RiskAssessmentModel
from smart_contract_deployer import SmartContractManager, ContractType
from cross_platform_compatibility import CrossPlatformManager, PlatformType
from qenex_core import QENEXCore, SecurityLevel
from blockchain import QENEXBlockchain
from secure_wallet import SecureWallet

logger = logging.getLogger(__name__)

# ============================================================================
# System Configuration
# ============================================================================

@dataclass
class SystemConfig:
    """QENEX system configuration"""
    instance_id: str
    environment: str  # development, staging, production
    debug_mode: bool
    enable_ai_ml: bool
    enable_blockchain: bool
    enable_smart_contracts: bool
    default_network: str
    security_level: str
    max_concurrent_transactions: int
    audit_logging: bool
    compliance_mode: str
    supported_protocols: List[str]
    geographic_region: str
    
@dataclass
class SystemStatus:
    """Current system status"""
    is_running: bool
    startup_time: datetime
    uptime_seconds: float
    processed_transactions: int
    active_connections: int
    memory_usage_mb: float
    cpu_usage_percent: float
    disk_usage_percent: float
    network_status: str
    security_status: str
    ai_status: str
    blockchain_status: str
    last_health_check: datetime

# ============================================================================
# QENEX Unified Banking System
# ============================================================================

class QENEXBankingSystem:
    """Main QENEX Banking Operating System"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.status = SystemStatus(
            is_running=False,
            startup_time=datetime.now(timezone.utc),
            uptime_seconds=0.0,
            processed_transactions=0,
            active_connections=0,
            memory_usage_mb=0.0,
            cpu_usage_percent=0.0,
            disk_usage_percent=0.0,
            network_status="initializing",
            security_status="initializing",
            ai_status="initializing",
            blockchain_status="initializing",
            last_health_check=datetime.now(timezone.utc)
        )
        
        # Core components
        self.cross_platform_manager: Optional[CrossPlatformManager] = None
        self.banking_protocols: Optional[BankingProtocolManager] = None
        self.ai_ml_system: Optional[SelfImprovingAI] = None
        self.smart_contracts: Optional[SmartContractManager] = None
        self.qenex_core: Optional[QENEXCore] = None
        self.blockchain: Optional[QENEXBlockchain] = None
        self.secure_wallet: Optional[SecureWallet] = None
        
        # System state
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.transaction_queue: asyncio.Queue = asyncio.Queue()
        self.event_handlers: Dict[str, List[callable]] = {}
        self.performance_metrics: Dict[str, Any] = {}
        
        logger.info(f"QENEX Banking System initialized with ID: {self.config.instance_id}")
        
    def _load_config(self, config_path: Optional[str]) -> SystemConfig:
        """Load system configuration"""
        
        default_config = SystemConfig(
            instance_id=f"qenex_{int(time.time())}",
            environment="development",
            debug_mode=True,
            enable_ai_ml=True,
            enable_blockchain=True,
            enable_smart_contracts=True,
            default_network="localhost",
            security_level="HIGH",
            max_concurrent_transactions=1000,
            audit_logging=True,
            compliance_mode="FULL",
            supported_protocols=["ISO20022", "SWIFT", "SEPA"],
            geographic_region="GLOBAL"
        )
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    # Update default config with loaded values
                    for key, value in config_data.items():
                        if hasattr(default_config, key):
                            setattr(default_config, key, value)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                
        return default_config
        
    async def initialize(self):
        """Initialize all system components"""
        
        logger.info("Initializing QENEX Banking System...")
        
        try:
            # Initialize cross-platform compatibility
            self.cross_platform_manager = CrossPlatformManager()
            await self.cross_platform_manager.initialize_platform_specific_features()
            logger.info("✓ Cross-platform compatibility initialized")
            
            # Initialize core components
            if self.config.enable_ai_ml:
                self.ai_ml_system = SelfImprovingAI()
                self.status.ai_status = "active"
                logger.info("✓ AI/ML system initialized")
            else:
                self.status.ai_status = "disabled"
                
            # Initialize banking protocols
            self.banking_protocols = BankingProtocolManager()
            logger.info("✓ Banking protocols initialized")
            
            # Initialize smart contracts
            if self.config.enable_smart_contracts:
                self.smart_contracts = SmartContractManager()
                logger.info("✓ Smart contract system initialized")
                
            # Initialize QENEX core
            self.qenex_core = QENEXCore()
            await self.qenex_core.initialize()
            logger.info("✓ QENEX core initialized")
            
            # Initialize blockchain
            if self.config.enable_blockchain:
                self.blockchain = QENEXBlockchain()
                await self.blockchain.initialize()
                self.status.blockchain_status = "active"
                logger.info("✓ Blockchain initialized")
            else:
                self.status.blockchain_status = "disabled"
                
            # Initialize secure wallet
            self.secure_wallet = SecureWallet()
            await self.secure_wallet.initialize()
            logger.info("✓ Secure wallet initialized")
            
            # Update system status
            self.status.is_running = True
            self.status.network_status = "active"
            self.status.security_status = "secure"
            self.status.last_health_check = datetime.now(timezone.utc)
            
            # Deploy core smart contracts if enabled
            if self.config.enable_smart_contracts and self.smart_contracts:
                await self._deploy_core_contracts()
                
            logger.info("🎉 QENEX Banking System fully initialized and ready!")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status.is_running = False
            raise
            
    async def _deploy_core_contracts(self):
        """Deploy core banking smart contracts"""
        
        try:
            logger.info("Deploying core banking smart contracts...")
            
            contracts = await self.smart_contracts.deploy_banking_suite(
                network=self.config.default_network,
                deploy_all=True
            )
            
            logger.info(f"Deployed {len(contracts)} core contracts")
            for contract_type, deployment in contracts.items():
                logger.info(f"- {contract_type}: {deployment.address}")
                
        except Exception as e:
            logger.error(f"Smart contract deployment failed: {e}")
            
    async def process_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a banking transaction through the unified system"""
        
        transaction_id = transaction_data.get('id', f"tx_{int(time.time())}")
        
        try:
            logger.info(f"Processing transaction {transaction_id}")
            
            # Step 1: AI/ML fraud detection and risk assessment
            ai_result = None
            if self.ai_ml_system:
                ai_result = self.ai_ml_system.process_transaction(transaction_data)
                
                # Block high-risk transactions
                if ai_result.get('final_decision') == 'REJECT':
                    return {
                        'transaction_id': transaction_id,
                        'status': 'REJECTED',
                        'reason': 'High fraud risk detected',
                        'ai_assessment': ai_result,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    
            # Step 2: Banking protocol processing
            protocol = transaction_data.get('protocol', 'ISO20022')
            protocol_result = None
            
            if self.banking_protocols and protocol in self.config.supported_protocols:
                protocol_result = await self.banking_protocols.process_payment(
                    protocol, transaction_data
                )
                
            # Step 3: Blockchain recording (if enabled)
            blockchain_result = None
            if self.blockchain and self.config.enable_blockchain:
                blockchain_result = await self.blockchain.add_transaction({
                    'id': transaction_id,
                    'type': 'banking_transaction',
                    'data': transaction_data,
                    'ai_assessment': ai_result,
                    'protocol_result': protocol_result
                })
                
            # Step 4: Smart contract execution (if applicable)
            smart_contract_result = None
            if (self.smart_contracts and 
                transaction_data.get('smart_contract_enabled', False)):
                
                # Execute payment processing contract
                smart_contract_result = await self._execute_payment_contract(
                    transaction_data, ai_result
                )
                
            # Step 5: Update system metrics
            self.status.processed_transactions += 1
            self._update_performance_metrics(transaction_id, time.time())
            
            # Compile final result
            result = {
                'transaction_id': transaction_id,
                'status': 'PROCESSED',
                'ai_assessment': ai_result,
                'protocol_result': protocol_result[:200] + '...' if protocol_result and len(protocol_result) > 200 else protocol_result,
                'blockchain_hash': blockchain_result.get('hash') if blockchain_result else None,
                'smart_contract_result': smart_contract_result,
                'processing_time_ms': (time.time() - float(transaction_id.split('_')[1])) * 1000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Transaction {transaction_id} processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Transaction {transaction_id} processing failed: {e}")
            return {
                'transaction_id': transaction_id,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
    async def _execute_payment_contract(self, transaction_data: Dict[str, Any], ai_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute payment processing smart contract"""
        
        try:
            # Get payment contract deployment
            deployments = self.smart_contracts.deployer.list_deployments()
            payment_contracts = [d for d in deployments if d.contract_type == ContractType.PAYMENT_PROCESSING]
            
            if not payment_contracts:
                return {'error': 'No payment processing contract deployed'}
                
            contract = payment_contracts[0]
            
            # Simulate contract execution
            return {
                'contract_address': contract.address,
                'transaction_hash': f"0x{hash(str(transaction_data)) % (2**64):016x}",
                'gas_used': 21000 + len(str(transaction_data)) * 68,
                'status': 'SUCCESS',
                'events': [
                    {
                        'event': 'PaymentInitiated',
                        'args': {
                            'sender': transaction_data.get('sender', 'unknown'),
                            'receiver': transaction_data.get('receiver', 'unknown'),
                            'amount': transaction_data.get('amount', 0)
                        }
                    }
                ]
            }
            
        except Exception as e:
            return {'error': str(e)}
            
    def _update_performance_metrics(self, transaction_id: str, start_time: float):
        """Update system performance metrics"""
        
        processing_time = time.time() - start_time
        
        if 'transaction_times' not in self.performance_metrics:
            self.performance_metrics['transaction_times'] = []
            
        self.performance_metrics['transaction_times'].append(processing_time)
        
        # Keep only recent metrics
        if len(self.performance_metrics['transaction_times']) > 1000:
            self.performance_metrics['transaction_times'] = \
                self.performance_metrics['transaction_times'][-1000:]
                
        # Calculate averages
        times = self.performance_metrics['transaction_times']
        self.performance_metrics['avg_processing_time'] = sum(times) / len(times)
        self.performance_metrics['max_processing_time'] = max(times)
        self.performance_metrics['min_processing_time'] = min(times)
        
    async def create_user_session(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user session"""
        
        session_id = f"session_{int(time.time())}_{user_data.get('user_id', 'anonymous')}"
        
        # Initialize secure wallet for user if needed
        if self.secure_wallet:
            wallet_address = await self.secure_wallet.create_wallet(user_data.get('user_id'))
        else:
            wallet_address = None
            
        session = {
            'session_id': session_id,
            'user_id': user_data.get('user_id'),
            'wallet_address': wallet_address,
            'created_at': datetime.now(timezone.utc),
            'last_activity': datetime.now(timezone.utc),
            'transaction_count': 0,
            'risk_profile': 'NORMAL'
        }
        
        self.active_sessions[session_id] = session
        self.status.active_connections += 1
        
        logger.info(f"Created session {session_id} for user {user_data.get('user_id')}")
        return session
        
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        current_time = datetime.now(timezone.utc)
        uptime = (current_time - self.status.startup_time).total_seconds()
        
        # Update runtime status
        self.status.uptime_seconds = uptime
        self.status.last_health_check = current_time
        
        # Get platform capabilities
        platform_info = {}
        if self.cross_platform_manager:
            platform_info = self.cross_platform_manager.get_platform_capabilities()
            
        # Get AI performance metrics
        ai_metrics = {}
        if self.ai_ml_system:
            ai_metrics = self.ai_ml_system.get_performance_metrics()
            
        # Get smart contract stats
        contract_stats = {}
        if self.smart_contracts:
            contract_stats = self.smart_contracts.get_contract_stats()
            
        return {
            'system_status': asdict(self.status),
            'configuration': asdict(self.config),
            'platform_info': platform_info,
            'performance_metrics': self.performance_metrics,
            'ai_metrics': ai_metrics,
            'contract_stats': contract_stats,
            'active_sessions': len(self.active_sessions),
            'components': {
                'cross_platform': bool(self.cross_platform_manager),
                'banking_protocols': bool(self.banking_protocols),
                'ai_ml_system': bool(self.ai_ml_system),
                'smart_contracts': bool(self.smart_contracts),
                'qenex_core': bool(self.qenex_core),
                'blockchain': bool(self.blockchain),
                'secure_wallet': bool(self.secure_wallet)
            }
        }
        
    async def shutdown(self):
        """Gracefully shutdown the system"""
        
        logger.info("Shutting down QENEX Banking System...")
        
        try:
            # Save system state
            await self._save_system_state()
            
            # Close active sessions
            for session_id in list(self.active_sessions.keys()):
                del self.active_sessions[session_id]
                
            # Shutdown components
            if self.blockchain:
                await self.blockchain.shutdown()
                
            if self.qenex_core:
                await self.qenex_core.shutdown()
                
            # Update status
            self.status.is_running = False
            self.status.active_connections = 0
            
            logger.info("QENEX Banking System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            
    async def _save_system_state(self):
        """Save current system state for recovery"""
        
        try:
            state = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'uptime_seconds': self.status.uptime_seconds,
                'processed_transactions': self.status.processed_transactions,
                'performance_metrics': self.performance_metrics,
                'config': asdict(self.config)
            }
            
            state_file = Path("qenex_system_state.json")
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
            logger.info("System state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")

# ============================================================================
# System Management CLI
# ============================================================================

class QENEXSystemManager:
    """Command-line interface for QENEX system management"""
    
    def __init__(self):
        self.system: Optional[QENEXBankingSystem] = None
        
    async def run_interactive_mode(self):
        """Run interactive management mode"""
        
        print("🏦 QENEX Banking Operating System Manager")
        print("=" * 45)
        
        while True:
            try:
                command = input("\nQENEX> ").strip().lower()
                
                if command in ['exit', 'quit', 'q']:
                    break
                elif command == 'init':
                    await self._initialize_system()
                elif command == 'status':
                    await self._show_status()
                elif command == 'test':
                    await self._run_test_transaction()
                elif command == 'session':
                    await self._create_test_session()
                elif command == 'help':
                    self._show_help()
                elif command == 'shutdown':
                    await self._shutdown_system()
                else:
                    print(f"Unknown command: {command}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                
        if self.system and self.system.status.is_running:
            await self.system.shutdown()
            
        print("\nGoodbye! 👋")
        
    async def _initialize_system(self):
        """Initialize the QENEX system"""
        
        if self.system and self.system.status.is_running:
            print("System is already initialized and running.")
            return
            
        print("Initializing QENEX Banking System...")
        self.system = QENEXBankingSystem()
        await self.system.initialize()
        print("✓ System initialized successfully!")
        
    async def _show_status(self):
        """Show system status"""
        
        if not self.system:
            print("System not initialized. Run 'init' first.")
            return
            
        status = await self.system.get_system_status()
        
        print(f"\n📊 QENEX System Status")
        print(f"- Running: {status['system_status']['is_running']}")
        print(f"- Uptime: {status['system_status']['uptime_seconds']:.1f} seconds")
        print(f"- Transactions Processed: {status['system_status']['processed_transactions']:,}")
        print(f"- Active Sessions: {status['active_sessions']}")
        print(f"- Platform: {status['platform_info'].get('platform', 'unknown')}")
        print(f"- Architecture: {status['platform_info'].get('architecture', 'unknown')}")
        
        print(f"\n🔧 Components Status:")
        for component, enabled in status['components'].items():
            status_icon = "✓" if enabled else "✗"
            print(f"- {component.replace('_', ' ').title()}: {status_icon}")
            
    async def _run_test_transaction(self):
        """Run a test transaction"""
        
        if not self.system or not self.system.status.is_running:
            print("System not initialized or not running. Run 'init' first.")
            return
            
        print("Running test transaction...")
        
        test_transaction = {
            'id': f'test_tx_{int(time.time())}',
            'user_id': 'test_user',
            'amount': 1000.50,
            'currency': 'USD',
            'sender': 'test_sender',
            'receiver': 'test_receiver',
            'protocol': 'ISO20022',
            'timestamp': time.time(),
            'merchant': 'grocery',
            'location': 'US'
        }
        
        result = await self.system.process_transaction(test_transaction)
        
        print(f"✓ Transaction processed: {result['status']}")
        print(f"- ID: {result['transaction_id']}")
        print(f"- AI Decision: {result.get('ai_assessment', {}).get('final_decision', 'N/A')}")
        if result.get('blockchain_hash'):
            print(f"- Blockchain Hash: {result['blockchain_hash']}")
            
    async def _create_test_session(self):
        """Create a test user session"""
        
        if not self.system or not self.system.status.is_running:
            print("System not initialized or not running. Run 'init' first.")
            return
            
        session = await self.system.create_user_session({
            'user_id': f'test_user_{int(time.time())}',
            'name': 'Test User',
            'email': 'ceo@qenex.ai'
        })
        
        print(f"✓ Session created: {session['session_id']}")
        if session.get('wallet_address'):
            print(f"- Wallet Address: {session['wallet_address']}")
            
    async def _shutdown_system(self):
        """Shutdown the system"""
        
        if not self.system or not self.system.status.is_running:
            print("System is not running.")
            return
            
        print("Shutting down system...")
        await self.system.shutdown()
        print("✓ System shutdown complete")
        
    def _show_help(self):
        """Show available commands"""
        
        print("\n📚 Available Commands:")
        print("- init      : Initialize QENEX Banking System")
        print("- status    : Show system status and metrics")
        print("- test      : Run a test transaction")
        print("- session   : Create a test user session")
        print("- shutdown  : Gracefully shutdown the system")
        print("- help      : Show this help message")
        print("- exit/quit : Exit the manager")

# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Main entry point for QENEX Banking System"""
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run automated demo
        print("🚀 QENEX Banking System - Automated Demo")
        print("=" * 40)
        
        # Initialize system
        system = QENEXBankingSystem()
        await system.initialize()
        
        # Show system status
        status = await system.get_system_status()
        print(f"\n📊 System initialized successfully!")
        print(f"- Platform: {status['platform_info'].get('platform', 'unknown')}")
        print(f"- Components: {sum(1 for v in status['components'].values() if v)}/{len(status['components'])}")
        
        # Process sample transactions
        print(f"\n💳 Processing sample transactions...")
        
        sample_transactions = [
            {
                'id': f'demo_tx_1',
                'user_id': 'alice',
                'amount': 250.00,
                'currency': 'EUR',
                'protocol': 'SEPA',
                'merchant': 'grocery',
                'location': 'DE'
            },
            {
                'id': f'demo_tx_2',
                'user_id': 'bob',
                'amount': 5000.00,
                'currency': 'USD',
                'protocol': 'SWIFT',
                'merchant': 'crypto',
                'location': 'NG'
            },
            {
                'id': f'demo_tx_3',
                'user_id': 'charlie',
                'amount': 75.50,
                'currency': 'USD',
                'protocol': 'ISO20022',
                'merchant': 'restaurant',
                'location': 'US'
            }
        ]
        
        for i, tx in enumerate(sample_transactions, 1):
            result = await system.process_transaction(tx)
            print(f"Transaction {i}: {result['status']} "
                  f"(AI: {result.get('ai_assessment', {}).get('final_decision', 'N/A')})")
                  
        # Show final status
        final_status = await system.get_system_status()
        print(f"\n📈 Demo completed!")
        print(f"- Transactions processed: {final_status['system_status']['processed_transactions']}")
        print(f"- Average processing time: {final_status['performance_metrics'].get('avg_processing_time', 0):.3f}s")
        
        await system.shutdown()
        
    else:
        # Run interactive manager
        manager = QENEXSystemManager()
        await manager.run_interactive_mode()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)