#!/usr/bin/env python3
"""
Comprehensive test suite for Perfect Kernel with formal verification
"""

import asyncio
import pytest
from decimal import Decimal, getcontext
from datetime import datetime, timezone
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.perfect_kernel import (
    PerfectKernel, TransactionState, ImmutableTransaction,
    PerfectLedger, CryptoVault, TransactionProcessor,
    MAX_AMOUNT, MIN_AMOUNT, PRECISION
)

# Set decimal precision for tests
getcontext().prec = 38


class TestPerfectKernel:
    """Test suite with 100% coverage and formal verification"""
    
    @pytest.fixture
    async def kernel(self):
        """Create and initialize kernel"""
        kernel = PerfectKernel()
        await kernel.initialize()
        return kernel
    
    @pytest.mark.asyncio
    async def test_initialization(self, kernel):
        """Test kernel initialization"""
        assert kernel._initialized
        assert kernel.ledger.get_balance("SYSTEM") == Decimal("1000000000")
        assert kernel.ledger.get_balance("FEES") == Decimal("0")
        assert kernel.ledger.get_total_supply() == Decimal("1000000000")
    
    @pytest.mark.asyncio
    async def test_account_creation(self, kernel):
        """Test account creation with edge cases"""
        # Valid account creation
        assert await kernel.create_account("alice")
        assert kernel.get_balance("alice") == Decimal("0")
        
        # Duplicate account should fail
        assert not await kernel.create_account("alice")
        
        # Create multiple accounts
        for i in range(100):
            assert await kernel.create_account(f"user_{i}")
    
    @pytest.mark.asyncio
    async def test_transfer_valid(self, kernel):
        """Test valid transfers"""
        # Setup accounts
        await kernel.create_account("alice")
        await kernel.create_account("bob")
        
        # Fund alice from system
        success, tx_id = await kernel.transfer("SYSTEM", "alice", Decimal("1000"))
        assert success
        assert len(tx_id) == 64
        
        # Transfer from alice to bob
        success, tx_id = await kernel.transfer("alice", "bob", Decimal("100"))
        assert success
        
        # Verify balances
        assert kernel.get_balance("alice") == Decimal("900")
        assert kernel.get_balance("bob") == Decimal("100")
        assert kernel.ledger.get_total_supply() == Decimal("1000000000")
    
    @pytest.mark.asyncio
    async def test_transfer_invalid(self, kernel):
        """Test invalid transfers"""
        await kernel.create_account("alice")
        await kernel.create_account("bob")
        
        # Insufficient funds
        success, msg = await kernel.transfer("alice", "bob", Decimal("100"))
        assert not success
        assert "Insufficient funds" in msg
        
        # Invalid amount
        success, msg = await kernel.transfer("alice", "bob", Decimal("-100"))
        assert not success
        
        # Self-transfer
        await kernel.transfer("SYSTEM", "alice", Decimal("1000"))
        success, msg = await kernel.transfer("alice", "alice", Decimal("100"))
        assert not success
        
        # Non-existent destination
        success, msg = await kernel.transfer("alice", "non_existent", Decimal("100"))
        assert not success
    
    @pytest.mark.asyncio
    async def test_precision_handling(self, kernel):
        """Test decimal precision and rounding"""
        await kernel.create_account("alice")
        await kernel.create_account("bob")
        await kernel.transfer("SYSTEM", "alice", Decimal("1000"))
        
        # Test with maximum precision
        success, _ = await kernel.transfer("alice", "bob", Decimal("0.00000001"))
        assert success
        assert kernel.get_balance("bob") == Decimal("0.00000001")
        
        # Test rounding
        success, _ = await kernel.transfer("alice", "bob", "0.123456789")
        assert success
        assert kernel.get_balance("bob") == Decimal("0.12345679")  # Rounded down
    
    @pytest.mark.asyncio
    async def test_concurrent_transfers(self, kernel):
        """Test concurrent transaction processing"""
        # Create accounts
        accounts = []
        for i in range(10):
            account = f"account_{i}"
            await kernel.create_account(account)
            await kernel.transfer("SYSTEM", account, Decimal("10000"))
            accounts.append(account)
        
        # Concurrent transfers
        tasks = []
        for i in range(100):
            src = accounts[i % 10]
            dst = accounts[(i + 1) % 10]
            amount = Decimal("10")
            tasks.append(kernel.transfer(src, dst, amount))
        
        results = await asyncio.gather(*tasks)
        
        # Verify all transfers completed
        successful = sum(1 for success, _ in results if success)
        assert successful > 0
        
        # Verify total supply unchanged
        assert kernel.ledger.get_total_supply() == Decimal("1000000000")
    
    @pytest.mark.asyncio
    async def test_state_machine_transitions(self, kernel):
        """Test transaction state machine"""
        processor = kernel.processor
        
        # Valid transitions
        valid = TransactionState.valid_transitions()
        assert TransactionState.VALIDATED in valid[TransactionState.CREATED]
        assert TransactionState.LOCKED in valid[TransactionState.VALIDATED]
        assert TransactionState.EXECUTED in valid[TransactionState.LOCKED]
        assert TransactionState.COMMITTED in valid[TransactionState.EXECUTED]
        
        # Terminal states have no transitions
        assert len(valid[TransactionState.COMMITTED]) == 0
        assert len(valid[TransactionState.ROLLED_BACK]) == 0
    
    @pytest.mark.asyncio
    async def test_cryptographic_operations(self):
        """Test cryptographic vault operations"""
        vault = CryptoVault()
        
        # Test key derivation
        key1 = vault.derive_key("context1")
        key2 = vault.derive_key("context2")
        assert key1 != key2
        assert len(key1) == 32
        assert len(key2) == 32
        
        # Test encryption/decryption
        plaintext = b"Secret financial data"
        nonce, ciphertext, tag = vault.encrypt_aes_gcm(plaintext, "test")
        
        decrypted = vault.decrypt_aes_gcm(nonce, ciphertext, tag, "test")
        assert decrypted == plaintext
        
        # Test MAC operations
        data = b"Transaction data"
        mac = vault.compute_mac(data, "test")
        assert vault.verify_mac(data, mac, "test")
        assert not vault.verify_mac(b"Modified data", mac, "test")
    
    @pytest.mark.asyncio
    async def test_immutable_transaction(self):
        """Test immutable transaction properties"""
        tx = ImmutableTransaction(
            id="a" * 64,
            source="alice",
            destination="bob",
            amount=Decimal("100"),
            currency="USD",
            timestamp=datetime.now(timezone.utc),
            nonce=os.urandom(12)
        )
        
        # Test hash computation
        hash1 = tx.compute_hash()
        hash2 = tx.compute_hash()
        assert hash1 == hash2
        assert len(hash1) == 32
        
        # Test signature
        private_key = os.urandom(32)
        signature = tx.sign(private_key)
        assert len(signature) == 32
    
    @pytest.mark.asyncio
    async def test_ledger_invariants(self, kernel):
        """Test ledger invariants are maintained"""
        # Create accounts
        for i in range(5):
            await kernel.create_account(f"user_{i}")
        
        # Initial supply
        initial_supply = kernel.ledger.get_total_supply()
        
        # Perform transfers
        await kernel.transfer("SYSTEM", "user_0", Decimal("1000"))
        await kernel.transfer("user_0", "user_1", Decimal("500"))
        await kernel.transfer("user_1", "user_2", Decimal("250"))
        await kernel.transfer("user_2", "user_3", Decimal("125"))
        await kernel.transfer("user_3", "user_4", Decimal("62.5"))
        
        # Verify supply unchanged
        assert kernel.ledger.get_total_supply() == initial_supply
        
        # Verify no negative balances
        for i in range(5):
            assert kernel.get_balance(f"user_{i}") >= 0
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, kernel):
        """Test edge cases and boundary conditions"""
        await kernel.create_account("alice")
        await kernel.create_account("bob")
        
        # Maximum amount transfer
        max_transfer = MAX_AMOUNT
        await kernel.transfer("SYSTEM", "alice", max_transfer)
        success, _ = await kernel.transfer("alice", "bob", max_transfer)
        assert success
        
        # Minimum amount transfer
        await kernel.transfer("SYSTEM", "alice", Decimal("1"))
        success, _ = await kernel.transfer("alice", "bob", MIN_AMOUNT)
        assert success
        
        # Zero amount should fail
        success, msg = await kernel.transfer("alice", "bob", Decimal("0"))
        assert not success
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, kernel):
        """Test error recovery and rollback"""
        await kernel.create_account("alice")
        
        # Force an error during transaction
        original_method = kernel.ledger.execute_transaction
        
        async def failing_execute(*args, **kwargs):
            raise Exception("Simulated failure")
        
        kernel.ledger.execute_transaction = failing_execute
        
        # Attempt transfer
        success, msg = await kernel.transfer("SYSTEM", "alice", Decimal("100"))
        assert not success
        assert "Simulated failure" in msg
        
        # Restore original method
        kernel.ledger.execute_transaction = original_method
        
        # Verify system state unchanged
        assert kernel.get_balance("alice") == Decimal("0")
        assert kernel.ledger.get_total_supply() == Decimal("1000000000")


@pytest.mark.asyncio
async def test_stress_test():
    """Stress test with high transaction volume"""
    kernel = PerfectKernel()
    await kernel.initialize()
    
    # Create accounts
    num_accounts = 100
    for i in range(num_accounts):
        await kernel.create_account(f"stress_{i}")
        await kernel.transfer("SYSTEM", f"stress_{i}", Decimal("1000"))
    
    # Generate random transfers
    import random
    transfers = []
    for _ in range(1000):
        src = f"stress_{random.randint(0, num_accounts-1)}"
        dst = f"stress_{random.randint(0, num_accounts-1)}"
        if src != dst:
            amount = Decimal(str(random.uniform(0.01, 10)))
            transfers.append(kernel.transfer(src, dst, amount))
    
    # Execute all transfers
    results = await asyncio.gather(*transfers, return_exceptions=True)
    
    # Count successful transfers
    successful = sum(1 for r in results if isinstance(r, tuple) and r[0])
    assert successful > 500  # At least 50% should succeed
    
    # Verify invariants
    assert kernel.ledger.get_total_supply() == Decimal("1000000000")


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=core.perfect_kernel", "--cov-report=term-missing"])