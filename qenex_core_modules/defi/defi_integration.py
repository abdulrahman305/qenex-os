#!/usr/bin/env python3
"""
QENEX OS DeFi Integration - Blockchain and DeFi protocol integration
"""

import asyncio
import hashlib
import json
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal
from threading import RLock, Lock
from contextlib import contextmanager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Token:
    """Thread-safe cryptocurrency token with atomic operations"""
    def __init__(self, symbol: str, name: str, decimals: int, address: str, balance: Decimal):
        self.symbol = symbol
        self.name = name
        self.decimals = decimals
        self.address = address
        self._balance = balance
        self._lock = RLock()
    
    @property 
    def balance(self) -> Decimal:
        """Get balance atomically"""
        with self._lock:
            return self._balance
    
    def update_balance(self, new_balance: Decimal) -> bool:
        """Update balance atomically"""
        with self._lock:
            if new_balance < 0:
                return False
            self._balance = new_balance
            return True
    
    def transfer_amount(self, amount: Decimal) -> bool:
        """Atomically deduct amount if sufficient balance exists"""
        with self._lock:
            if self._balance >= amount:
                self._balance -= amount
                return True
            return False
    
    def add_amount(self, amount: Decimal):
        """Atomically add amount to balance"""
        with self._lock:
            self._balance += amount

@dataclass
class Transaction:
    """Represents a blockchain transaction"""
    tx_hash: str
    from_addr: str
    to_addr: str
    value: Decimal
    gas_price: Decimal
    status: str
    timestamp: float

class AtomicStakingPosition:
    """Thread-safe staking position"""
    def __init__(self, token_symbol: str):
        self.token_symbol = token_symbol
        self._amount = Decimal("0")
        self._lock = RLock()
    
    @property
    def amount(self) -> Decimal:
        with self._lock:
            return self._amount
    
    def add_stake(self, amount: Decimal) -> bool:
        with self._lock:
            if amount <= 0:
                return False
            self._amount += amount
            return True
    
    def remove_stake(self, amount: Decimal) -> bool:
        with self._lock:
            if self._amount >= amount:
                self._amount -= amount
                return True
            return False

class AtomicLiquidityPool:
    """Thread-safe liquidity pool"""
    def __init__(self, token_a: str, token_b: str):
        self.token_a = token_a
        self.token_b = token_b
        self._reserve_a = Decimal("0")
        self._reserve_b = Decimal("0")
        self._user_share = Decimal("0")
        self._lock = RLock()
    
    @contextmanager
    def atomic_operation(self):
        """Context manager for atomic pool operations"""
        self._lock.acquire()
        try:
            yield self
        finally:
            self._lock.release()
    
    def add_liquidity(self, amount_a: Decimal, amount_b: Decimal) -> Decimal:
        """Add liquidity and return LP tokens earned"""
        with self._lock:
            lp_tokens = (amount_a * amount_b) ** Decimal('0.5')
            self._reserve_a += amount_a
            self._reserve_b += amount_b
            self._user_share += lp_tokens
            return lp_tokens
    
    def get_reserves(self) -> tuple[Decimal, Decimal]:
        with self._lock:
            return self._reserve_a, self._reserve_b

class DeFiIntegration:
    """Thread-safe DeFi protocol integration manager"""
    
    def __init__(self):
        self.wallet_address: Optional[str] = None
        self.private_key: Optional[str] = None
        self.tokens: Dict[str, Token] = {}
        self.transactions: List[Transaction] = []
        self.staking_positions: Dict[str, AtomicStakingPosition] = {}
        self.liquidity_pools: Dict[str, AtomicLiquidityPool] = {}
        self.initialized = False
        
        # Thread safety locks
        self._wallet_lock = RLock()
        self._transactions_lock = RLock()
        self._tokens_lock = RLock()
        self._staking_lock = RLock()
        self._pools_lock = RLock()
        
    async def initialize(self):
        """Initialize DeFi integration"""
        print("💰 Initializing DeFi Integration...")
        
        # Initialize default wallet
        self._create_wallet()
        
        # Load default tokens
        self._load_default_tokens()
        
        # Start DeFi services
        asyncio.create_task(self.price_monitor())
        asyncio.create_task(self.yield_optimizer())
        
        self.initialized = True
        print("✅ DeFi Integration initialized")
    
    async def shutdown(self):
        """Shutdown DeFi integration"""
        self.initialized = False
        print("💰 DeFi Integration shutdown")
    
    def _create_wallet(self):
        """Create a new wallet"""
        # Use environment variable or secure key management system
        import os
        private_key_env = os.environ.get('QENEX_PRIVATE_KEY')
        if private_key_env:
            # Use key from secure environment variable
            self.private_key = private_key_env
        else:
            # Generate temporary key for demo purposes only - NOT for production
            import secrets
            self.private_key = "0x" + secrets.token_hex(32)
            print("WARNING: Using demo key - set QENEX_PRIVATE_KEY env variable for production")
        
        # Never log private keys in production
        self.wallet_address = "0x" + hashlib.sha256(self.private_key.encode()).hexdigest()[:40]
        print(f"🔑 Wallet initialized: {self.wallet_address[:20]}...")
    
    def _load_default_tokens(self):
        """Load default tokens"""
        self.tokens = {
            "QXC": Token(
                symbol="QXC",
                name="QENEX Coin",
                decimals=18,
                address="0x" + "0" * 40,
                balance=Decimal("1000.0")
            ),
            "ETH": Token(
                symbol="ETH",
                name="Ethereum",
                decimals=18,
                address="0x" + "0" * 40,
                balance=Decimal("0.5")
            ),
            "USDC": Token(
                symbol="USDC",
                name="USD Coin",
                decimals=6,
                address="0x" + "0" * 40,
                balance=Decimal("100.0")
            )
        }
    
    async def get_balance(self, token_symbol: str = "QXC") -> Decimal:
        """Get token balance"""
        if token_symbol in self.tokens:
            return self.tokens[token_symbol].balance
        return Decimal("0")
    
    async def transfer(self, to_address: str, amount: Decimal, token_symbol: str = "QXC") -> Optional[str]:
        """Transfer tokens atomically"""
        with self._tokens_lock:
            if token_symbol not in self.tokens:
                logger.error(f"Token {token_symbol} not found")
                return None
            
            token = self.tokens[token_symbol]
        
        # Check and atomically deduct balance
        if not token.transfer_amount(amount):
            logger.error(f"Insufficient balance for {token_symbol}: requested {amount}, available {token.balance}")
            return None
        
        try:
            # Create transaction
            tx_hash = hashlib.sha256(f"{self.wallet_address}{to_address}{amount}{time.time()}".encode()).hexdigest()
            
            transaction = Transaction(
                tx_hash=tx_hash,
                from_addr=self.wallet_address,
                to_addr=to_address,
                value=amount,
                gas_price=Decimal("20"),  # Gwei
                status="pending",
                timestamp=time.time()
            )
            
            # Add transaction to list atomically
            with self._transactions_lock:
                self.transactions.append(transaction)
            
            # Simulate transaction processing
            await asyncio.sleep(0.1)  # Reduced for better UX
            
            # Mark transaction as confirmed
            transaction.status = "confirmed"
            
            logger.info(f"Transferred {amount} {token_symbol} to {to_address[:20]}... (tx: {tx_hash[:16]}...)")
            return tx_hash
            
        except Exception as e:
            # Rollback balance change if transaction creation failed
            token.add_amount(amount)
            logger.error(f"Transfer failed: {e}")
            return None
    
    async def stake(self, amount: Decimal, token_symbol: str = "QXC") -> bool:
        """Stake tokens atomically"""
        with self._tokens_lock:
            if token_symbol not in self.tokens:
                logger.error(f"Token {token_symbol} not found")
                return False
            
            token = self.tokens[token_symbol]
        
        # Atomically check and deduct balance
        if not token.transfer_amount(amount):
            logger.error(f"Insufficient balance for staking: requested {amount}, available {token.balance}")
            return False
        
        try:
            # Ensure staking position exists
            with self._staking_lock:
                if token_symbol not in self.staking_positions:
                    self.staking_positions[token_symbol] = AtomicStakingPosition(token_symbol)
                
                staking_position = self.staking_positions[token_symbol]
            
            # Add to staking position atomically
            if staking_position.add_stake(amount):
                logger.info(f"Staked {amount} {token_symbol} (total staked: {staking_position.amount})")
                return True
            else:
                # Rollback if staking failed
                token.add_amount(amount)
                return False
                
        except Exception as e:
            # Rollback balance change if staking failed
            token.add_amount(amount)
            logger.error(f"Staking failed: {e}")
            return False
    
    async def unstake(self, amount: Decimal, token_symbol: str = "QXC") -> bool:
        """Unstake tokens atomically with rewards"""
        with self._staking_lock:
            if token_symbol not in self.staking_positions:
                logger.error(f"No staking position found for {token_symbol}")
                return False
            
            staking_position = self.staking_positions[token_symbol]
        
        # Check and atomically remove from staking
        if not staking_position.remove_stake(amount):
            logger.error(f"Insufficient staked balance: requested {amount}, available {staking_position.amount}")
            return False
        
        try:
            with self._tokens_lock:
                if token_symbol not in self.tokens:
                    logger.error(f"Token {token_symbol} not found")
                    # Rollback staking removal
                    staking_position.add_stake(amount)
                    return False
                
                token = self.tokens[token_symbol]
            
            # Calculate rewards (10% APY, assuming 1 day staking for simplicity)
            rewards = amount * Decimal("0.1") / Decimal("365")
            
            # Return staked amount plus rewards to token balance
            token.add_amount(amount + rewards)
            
            logger.info(f"Unstaked {amount} {token_symbol} with {rewards:.6f} rewards "
                       f"(remaining staked: {staking_position.amount})")
            return True
            
        except Exception as e:
            # Rollback staking removal if token update failed
            staking_position.add_stake(amount)
            logger.error(f"Unstaking failed: {e}")
            return False
    
    async def add_liquidity(self, token_a: str, token_b: str, amount_a: Decimal, amount_b: Decimal) -> Optional[str]:
        """Add liquidity to a pool"""
        if token_a not in self.tokens or token_b not in self.tokens:
            return None
        
        if self.tokens[token_a].balance < amount_a or self.tokens[token_b].balance < amount_b:
            print("❌ Insufficient balance for liquidity")
            return None
        
        # Create pool ID
        pool_id = f"{token_a}-{token_b}"
        
        # Deduct from balance
        self.tokens[token_a].balance -= amount_a
        self.tokens[token_b].balance -= amount_b
        
        # Add to pool
        if pool_id not in self.liquidity_pools:
            self.liquidity_pools[pool_id] = {
                "token_a": token_a,
                "token_b": token_b,
                "reserve_a": Decimal("0"),
                "reserve_b": Decimal("0"),
                "user_share": Decimal("0")
            }
        
        pool = self.liquidity_pools[pool_id]
        pool["reserve_a"] += amount_a
        pool["reserve_b"] += amount_b
        # Simplified LP token calculation using approximation
        pool["user_share"] += (amount_a * amount_b) ** Decimal('0.5')
        
        print(f"✅ Added liquidity to {pool_id} pool")
        return pool_id
    
    async def swap(self, token_in: str, token_out: str, amount_in: Decimal) -> Optional[Decimal]:
        """Swap tokens"""
        pool_id = f"{token_in}-{token_out}"
        reverse_pool_id = f"{token_out}-{token_in}"
        
        pool = self.liquidity_pools.get(pool_id) or self.liquidity_pools.get(reverse_pool_id)
        
        if not pool:
            print(f"❌ No liquidity pool for {token_in}-{token_out}")
            return None
        
        if self.tokens[token_in].balance < amount_in:
            print(f"❌ Insufficient {token_in} balance")
            return None
        
        # Calculate output amount (constant product formula)
        if pool["token_a"] == token_in:
            reserve_in = pool["reserve_a"]
            reserve_out = pool["reserve_b"]
        else:
            reserve_in = pool["reserve_b"]
            reserve_out = pool["reserve_a"]
        
        amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
        fee = amount_out * Decimal("0.003")  # 0.3% fee
        amount_out -= fee
        
        # Execute swap
        self.tokens[token_in].balance -= amount_in
        self.tokens[token_out].balance += amount_out
        
        # Update reserves
        if pool["token_a"] == token_in:
            pool["reserve_a"] += amount_in
            pool["reserve_b"] -= amount_out
        else:
            pool["reserve_b"] += amount_in
            pool["reserve_a"] -= amount_out
        
        print(f"✅ Swapped {amount_in} {token_in} for {amount_out:.4f} {token_out}")
        return amount_out
    
    async def get_price(self, token_symbol: str) -> Decimal:
        """Get token price in USD"""
        # Simulated prices
        prices = {
            "QXC": Decimal("0.50"),
            "ETH": Decimal("2000.00"),
            "USDC": Decimal("1.00")
        }
        
        return prices.get(token_symbol, Decimal("0"))
    
    async def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value in USD"""
        total = Decimal("0")
        
        for symbol, token in self.tokens.items():
            price = await self.get_price(symbol)
            total += token.balance * price
        
        for symbol, amount in self.staking_positions.items():
            price = await self.get_price(symbol)
            total += amount * price
        
        return total
    
    async def price_monitor(self):
        """Monitor token prices"""
        while self.initialized:
            await asyncio.sleep(60)
            
            # Simulate price updates
            for symbol in self.tokens:
                price = await self.get_price(symbol)
                # Add small random variation
                import random
                variation = Decimal(str(random.uniform(-0.05, 0.05)))
                new_price = price * (Decimal("1") + variation)
                print(f"📈 {symbol} price: ${new_price:.2f} ({variation*100:+.2f}%)")
    
    async def yield_optimizer(self):
        """Optimize yield farming strategies"""
        while self.initialized:
            await asyncio.sleep(120)
            
            # Calculate staking rewards
            for symbol, amount in self.staking_positions.items():
                if amount > 0:
                    # 10% APY, calculated per 2 minutes for demo
                    rewards = amount * Decimal("0.1") / Decimal("365") / Decimal("720")
                    self.tokens[symbol].balance += rewards
                    print(f"💎 Staking rewards: {rewards:.6f} {symbol}")
    
    def get_status(self) -> Dict:
        """Get DeFi integration status"""
        portfolio_value = Decimal("0")
        try:
            # Calculate synchronously for status
            for symbol, token in self.tokens.items():
                portfolio_value += token.balance * Decimal("0.50")  # Simplified
        except:
            pass
        
        return {
            "status": "active" if self.initialized else "inactive",
            "wallet_connected": self.wallet_address is not None,
            "wallet_address": self.wallet_address[:10] + "..." if self.wallet_address else None,
            "tokens": len(self.tokens),
            "staking_positions": len(self.staking_positions),
            "liquidity_pools": len(self.liquidity_pools),
            "portfolio_value": float(portfolio_value),
            "transactions": len(self.transactions)
        }

# Singleton instance
defi_integration = DeFiIntegration()

async def main():
    """Main function for testing"""
    await defi_integration.initialize()
    
    # Check balance
    balance = await defi_integration.get_balance("QXC")
    print(f"QXC Balance: {balance}")
    
    # Transfer tokens
    tx_hash = await defi_integration.transfer("0x123...456", Decimal("10"), "QXC")
    print(f"Transaction: {tx_hash[:16]}...")
    
    # Stake tokens
    await defi_integration.stake(Decimal("100"), "QXC")
    
    # Add liquidity
    pool_id = await defi_integration.add_liquidity("QXC", "USDC", Decimal("50"), Decimal("25"))
    
    # Swap tokens
    amount_out = await defi_integration.swap("USDC", "QXC", Decimal("10"))
    
    # Check portfolio
    portfolio = await defi_integration.get_portfolio_value()
    print(f"Portfolio value: ${portfolio:.2f}")
    
    # Get status
    status = defi_integration.get_status()
    print(f"DeFi status: {json.dumps(status, indent=2)}")
    
    # Wait for some rewards
    await asyncio.sleep(3)
    
    # Unstake
    await defi_integration.unstake(Decimal("50"), "QXC")
    
    await defi_integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())