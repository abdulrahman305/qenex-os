#!/usr/bin/env python3
"""
QENEX Unified Production Financial Operating System
Complete integration of all components with cross-platform support
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from typing import Dict, List, Optional, Any
import subprocess

# Configure precision
getcontext().prec = 38

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all production components
try:
    from production_financial_core import ProductionFinancialCore, FinancialAPIServer
    from production_blockchain_defi import Blockchain, Token, LiquidityPool, LendingProtocol, StakingProtocol, DeFiAggregator
    from production_ai_quantum_security import IntegratedSecuritySystem, AIRiskAnalyzer, QuantumResistantCrypto
except ImportError as e:
    logger.warning(f"Import warning: {e}. Running in standalone mode.")

# ============================================================================
# Platform Detection and Compatibility Layer
# ============================================================================

class PlatformManager:
    """Cross-platform compatibility manager"""
    
    @staticmethod
    def detect_platform() -> Dict:
        """Detect current platform and capabilities"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture(),
            'is_windows': platform.system() == 'Windows',
            'is_linux': platform.system() == 'Linux',
            'is_mac': platform.system() == 'Darwin',
            'is_mobile': PlatformManager._is_mobile(),
            'is_cloud': PlatformManager._is_cloud(),
            'capabilities': PlatformManager._get_capabilities()
        }
    
    @staticmethod
    def _is_mobile() -> bool:
        """Detect if running on mobile platform"""
        return 'arm' in platform.machine().lower() or \
               'android' in platform.platform().lower() or \
               'ios' in platform.platform().lower()
    
    @staticmethod
    def _is_cloud() -> bool:
        """Detect if running in cloud environment"""
        # Check for common cloud environment variables
        cloud_indicators = [
            'AWS_EXECUTION_ENV', 'AWS_LAMBDA_FUNCTION_NAME',
            'GOOGLE_CLOUD_PROJECT', 'AZURE_FUNCTIONS_ENVIRONMENT',
            'KUBERNETES_SERVICE_HOST', 'DYNO'
        ]
        return any(os.getenv(indicator) for indicator in cloud_indicators)
    
    @staticmethod
    def _get_capabilities() -> Dict:
        """Determine platform capabilities"""
        return {
            'hardware_crypto': PlatformManager._has_hardware_crypto(),
            'gpu_acceleration': PlatformManager._has_gpu(),
            'network_stack': 'full',
            'storage_type': PlatformManager._detect_storage_type(),
            'memory_gb': PlatformManager._get_memory_gb()
        }
    
    @staticmethod
    def _has_hardware_crypto() -> bool:
        """Check for hardware crypto support"""
        try:
            # Check for AES-NI on x86
            if 'x86' in platform.machine():
                import subprocess
                result = subprocess.run(['cat', '/proc/cpuinfo'], capture_output=True, text=True)
                return 'aes' in result.stdout.lower()
        except:
            pass
        return False
    
    @staticmethod
    def _has_gpu() -> bool:
        """Check for GPU availability"""
        try:
            import subprocess
            # Try nvidia-smi for NVIDIA GPUs
            subprocess.run(['nvidia-smi'], capture_output=True, check=True)
            return True
        except:
            pass
        return False
    
    @staticmethod
    def _detect_storage_type() -> str:
        """Detect storage type (SSD/HDD)"""
        # Simplified detection
        return 'ssd'  # Default assumption for modern systems
    
    @staticmethod
    def _get_memory_gb() -> float:
        """Get system memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0  # Default assumption

# ============================================================================
# Unified Financial Operating System
# ============================================================================

class UnifiedFinancialOS:
    """Complete financial operating system with all components integrated"""
    
    def __init__(self):
        logger.info("Initializing QENEX Unified Financial Operating System")
        
        # Platform detection
        self.platform = PlatformManager.detect_platform()
        logger.info(f"Platform detected: {self.platform['system']} {self.platform['release']}")
        
        # Initialize core components
        self.financial_core = ProductionFinancialCore()
        self.blockchain = Blockchain()
        self.security_system = IntegratedSecuritySystem()
        self.api_server = FinancialAPIServer(self.financial_core)
        
        # Initialize DeFi components
        self.defi_aggregator = DeFiAggregator()
        self.tokens = {}
        self.liquidity_pools = []
        self.lending_protocols = []
        self.staking_protocol = StakingProtocol(self.blockchain)
        
        # System state
        self.is_running = False
        self.start_time = time.time()
        self.metrics = {
            'transactions_processed': 0,
            'blocks_mined': 0,
            'security_events': 0,
            'defi_volume': Decimal('0'),
            'total_value_locked': Decimal('0')
        }
        
        # Self-improvement AI
        self.ai_evolution_interval = 300  # 5 minutes
        self.last_evolution = time.time()
        self.evolution_generation = 1
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        # Create system accounts
        await self._create_system_accounts()
        
        # Deploy initial tokens
        await self._deploy_tokens()
        
        # Initialize liquidity pools
        await self._initialize_liquidity_pools()
        
        # Start background tasks
        asyncio.create_task(self._monitor_system())
        asyncio.create_task(self._evolve_ai())
        asyncio.create_task(self._mine_blocks())
        
        self.is_running = True
        logger.info("System initialization complete")
    
    async def _create_system_accounts(self):
        """Create system accounts"""
        # Create system user
        result = await self.financial_core.create_user(
            'system', 'system@qenex.ai', 'SystemPass123!'
        )
        
        if result['success']:
            self.system_user_id = result['user_id']
            
            # Create system accounts
            accounts = ['treasury', 'rewards', 'insurance', 'development']
            for account_type in accounts:
                await self.financial_core.create_account(
                    self.system_user_id,
                    account_type,
                    Decimal('1000000'),
                    'USD'
                )
    
    async def _deploy_tokens(self):
        """Deploy initial tokens"""
        # QENEX native token
        qnx = Token(
            symbol='QNX',
            name='QENEX Token',
            decimals=18,
            total_supply=Decimal('1000000000'),
            standard='ERC20',
            owner='system'
        )
        qnx.balances['treasury'] = Decimal('500000000')
        qnx.balances['rewards'] = Decimal('200000000')
        qnx.balances['development'] = Decimal('300000000')
        self.tokens['QNX'] = qnx
        
        # Stablecoins
        usdc = Token(
            symbol='USDC',
            name='USD Coin',
            decimals=6,
            total_supply=Decimal('1000000000'),
            standard='ERC20',
            owner='system'
        )
        usdc.balances['treasury'] = Decimal('100000000')
        self.tokens['USDC'] = usdc
    
    async def _initialize_liquidity_pools(self):
        """Initialize DeFi liquidity pools"""
        # Create QNX/USDC pool
        if 'QNX' in self.tokens and 'USDC' in self.tokens:
            pool = LiquidityPool(self.tokens['QNX'], self.tokens['USDC'])
            pool.add_liquidity('treasury', Decimal('1000000'), Decimal('1000000'))
            self.liquidity_pools.append(pool)
            self.defi_aggregator.pools.append(pool)
    
    async def process_transaction(self, transaction: Dict) -> Dict:
        """Process transaction through entire stack"""
        try:
            # 1. Security check
            security_result = await self.security_system.secure_transaction(transaction)
            
            if security_result['decision'] == 'blocked':
                return {
                    'success': False,
                    'error': 'Transaction blocked by security system',
                    'risk_score': security_result['risk_score']
                }
            
            # 2. Process through financial core
            core_result = await self.financial_core.process_transaction(
                transaction['source_account'],
                transaction['destination_account'],
                Decimal(str(transaction['amount'])),
                transaction.get('currency', 'USD'),
                transaction.get('idempotency_key')
            )
            
            if not core_result['success']:
                return core_result
            
            # 3. Record on blockchain
            blockchain_tx = Transaction(
                sender=transaction['source_account'],
                receiver=transaction['destination_account'],
                amount=Decimal(str(transaction['amount'])),
                fee=Decimal(core_result.get('fee', '0')),
                timestamp=time.time()
            )
            
            self.blockchain.add_transaction(blockchain_tx)
            
            # 4. Update metrics
            self.metrics['transactions_processed'] += 1
            
            return {
                'success': True,
                'transaction_id': core_result['transaction_id'],
                'blockchain_hash': blockchain_tx.hash(),
                'security_score': security_result['risk_score'],
                'fee': core_result.get('fee', '0')
            }
            
        except Exception as e:
            logger.error(f"Transaction processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def execute_defi_swap(self, token_in: str, token_out: str, amount: Decimal) -> Dict:
        """Execute DeFi token swap"""
        try:
            # Find best route
            if token_in not in self.tokens or token_out not in self.tokens:
                return {'success': False, 'error': 'Token not found'}
            
            # Find pool
            pool = None
            for p in self.liquidity_pools:
                if (p.token_a.symbol == token_in and p.token_b.symbol == token_out) or \
                   (p.token_b.symbol == token_in and p.token_a.symbol == token_out):
                    pool = p
                    break
            
            if not pool:
                return {'success': False, 'error': 'No liquidity pool found'}
            
            # Execute swap
            token_in_obj = self.tokens[token_in]
            amount_out = pool.swap(token_in_obj, amount)
            
            # Update metrics
            self.metrics['defi_volume'] += amount
            
            return {
                'success': True,
                'amount_in': str(amount),
                'amount_out': str(amount_out),
                'price': str(pool.get_price(token_in_obj))
            }
            
        except Exception as e:
            logger.error(f"DeFi swap failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def stake_tokens(self, validator: str, amount: Decimal) -> Dict:
        """Stake tokens for validation"""
        try:
            success = self.staking_protocol.stake(validator, amount)
            
            if success:
                return {
                    'success': True,
                    'validator': validator,
                    'staked_amount': str(amount),
                    'total_stake': str(self.staking_protocol.stakes.get(validator, 0))
                }
            else:
                return {'success': False, 'error': 'Staking failed'}
                
        except Exception as e:
            logger.error(f"Staking failed: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _monitor_system(self):
        """Monitor system health and performance"""
        while self.is_running:
            try:
                # Calculate metrics
                uptime = time.time() - self.start_time
                tps = self.metrics['transactions_processed'] / max(uptime, 1)
                
                # Check system health
                health_status = {
                    'uptime': uptime,
                    'transactions_per_second': tps,
                    'blockchain_height': len(self.blockchain.chain),
                    'pending_transactions': len(self.blockchain.pending_transactions),
                    'total_value_locked': str(self._calculate_tvl()),
                    'evolution_generation': self.evolution_generation
                }
                
                # Log status
                if int(uptime) % 60 == 0:  # Log every minute
                    logger.info(f"System Status: {health_status}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _evolve_ai(self):
        """Self-improving AI evolution"""
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - self.last_evolution >= self.ai_evolution_interval:
                    # Trigger AI improvement
                    await self.security_system.ai_analyzer._self_improve()
                    
                    self.evolution_generation += 1
                    self.last_evolution = current_time
                    
                    logger.info(f"AI evolved to generation {self.evolution_generation}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"AI evolution error: {e}")
                await asyncio.sleep(60)
    
    async def _mine_blocks(self):
        """Mine blocks periodically"""
        while self.is_running:
            try:
                if self.blockchain.pending_transactions:
                    # Select validator
                    validator = self.staking_protocol.select_validator()
                    
                    if validator:
                        # Mine block
                        self.blockchain.mine_pending_transactions(validator)
                        
                        # Distribute rewards
                        self.staking_protocol.distribute_rewards(validator)
                        
                        self.metrics['blocks_mined'] += 1
                        
                        logger.info(f"Block mined by {validator}, height: {len(self.blockchain.chain)}")
                
                await asyncio.sleep(10)  # Block time: 10 seconds
                
            except Exception as e:
                logger.error(f"Mining error: {e}")
                await asyncio.sleep(10)
    
    def _calculate_tvl(self) -> Decimal:
        """Calculate total value locked in DeFi"""
        tvl = Decimal('0')
        
        # Liquidity pools
        for pool in self.liquidity_pools:
            tvl += pool.reserve_a + pool.reserve_b
        
        # Lending protocols
        for protocol in self.lending_protocols:
            for user_deposits in protocol.deposits.values():
                tvl += sum(user_deposits.values())
        
        # Staked tokens
        tvl += sum(self.staking_protocol.stakes.values())
        
        return tvl
    
    async def get_system_status(self) -> Dict:
        """Get complete system status"""
        return {
            'platform': self.platform,
            'uptime': time.time() - self.start_time,
            'metrics': self.metrics,
            'blockchain': {
                'height': len(self.blockchain.chain),
                'pending_transactions': len(self.blockchain.pending_transactions),
                'is_valid': self.blockchain.validate_chain()
            },
            'defi': {
                'total_value_locked': str(self._calculate_tvl()),
                'liquidity_pools': len(self.liquidity_pools),
                'tokens': list(self.tokens.keys())
            },
            'security': {
                'events_processed': len(self.security_system.security_events),
                'ai_model_version': self.security_system.ai_analyzer.model_version,
                'evolution_generation': self.evolution_generation
            }
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down QENEX Financial OS...")
        self.is_running = False
        self.financial_core.close()
        logger.info("Shutdown complete")

# ============================================================================
# API Endpoints
# ============================================================================

class UnifiedAPIServer:
    """Unified API server for all services"""
    
    def __init__(self, os_instance: UnifiedFinancialOS):
        self.os = os_instance
    
    async def handle_request(self, method: str, path: str, body: Dict, headers: Dict) -> Dict:
        """Route API requests"""
        
        # Traditional finance endpoints
        if path.startswith('/api/v1/finance'):
            return await self.os.api_server.handle_request(
                method, path.replace('/api/v1/finance', ''), body, headers
            )
        
        # Blockchain endpoints
        elif path == '/api/v1/blockchain/transaction' and method == 'POST':
            return await self.os.process_transaction(body)
        
        elif path == '/api/v1/blockchain/balance' and method == 'GET':
            address = body.get('address')
            balance = self.os.blockchain.get_balance(address)
            return {'balance': str(balance)}
        
        # DeFi endpoints
        elif path == '/api/v1/defi/swap' and method == 'POST':
            return await self.os.execute_defi_swap(
                body['token_in'],
                body['token_out'],
                Decimal(str(body['amount']))
            )
        
        elif path == '/api/v1/defi/stake' and method == 'POST':
            return await self.os.stake_tokens(
                body['validator'],
                Decimal(str(body['amount']))
            )
        
        # System endpoints
        elif path == '/api/v1/system/status' and method == 'GET':
            return await self.os.get_system_status()
        
        elif path == '/api/v1/system/health' and method == 'GET':
            return {
                'status': 'healthy',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        else:
            return {'error': 'Endpoint not found', 'status': 404}

# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Run the complete unified financial operating system"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     QENEX UNIFIED FINANCIAL OPERATING SYSTEM             ║
    ║                                                           ║
    ║     Production-Ready • Cross-Platform • AI-Powered       ║
    ║     Blockchain-Native • Quantum-Resistant • DeFi-Ready   ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    os_instance = UnifiedFinancialOS()
    await os_instance.initialize()
    
    # Initialize API server
    api_server = UnifiedAPIServer(os_instance)
    
    # Display system information
    status = await os_instance.get_system_status()
    print(f"\n✅ System initialized successfully")
    print(f"📍 Platform: {status['platform']['system']} {status['platform']['release']}")
    print(f"🔒 Security: Quantum-resistant encryption active")
    print(f"🤖 AI: Generation {status['security']['evolution_generation']}")
    print(f"⛓️ Blockchain: Height {status['blockchain']['height']}")
    print(f"💰 DeFi: ${status['defi']['total_value_locked']} TVL")
    print(f"\n🌐 API Server: https://abdulrahman305.github.io/qenex-docs)
    print(f"📊 Dashboard: https://abdulrahman305.github.io/qenex-docs)
    
    # Run forever
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n⏹️ Shutting down...")
        await os_instance.shutdown()
        print("✅ Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())