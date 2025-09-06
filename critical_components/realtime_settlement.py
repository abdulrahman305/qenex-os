#!/usr/bin/env python3
"""
Real-time Settlement Engine with Atomic Cross-border Payments
Implements RTGS with multi-currency support and guaranteed finality
"""

import asyncio
import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_DOWN
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, AsyncIterator
import json
import struct
import threading
from queue import PriorityQueue
import logging
from datetime import datetime, timedelta
import redis
import msgpack

# High precision for financial calculations
getcontext().prec = 50
getcontext().rounding = ROUND_DOWN

# Settlement Configuration
SETTLEMENT_BATCH_SIZE = 1000
SETTLEMENT_INTERVAL_MS = 100  # 100ms settlement cycles
NETTING_WINDOW_MS = 5000  # 5 second netting window
LIQUIDITY_BUFFER_RATIO = Decimal('0.1')  # 10% liquidity buffer
MAX_SETTLEMENT_RETRIES = 3
ATOMIC_TIMEOUT_MS = 10000  # 10 second timeout for atomic operations

# Cross-border Configuration
CORRESPONDENT_TIMEOUT_MS = 30000
FX_RATE_UPDATE_INTERVAL_MS = 1000
NOSTRO_RECONCILIATION_INTERVAL_MS = 60000

# Compliance thresholds
MAX_SINGLE_TRANSACTION = Decimal('10000000')  # $10M
DAILY_LIMIT_PER_ACCOUNT = Decimal('100000000')  # $100M
SUSPICIOUS_PATTERN_THRESHOLD = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SettlementStatus(Enum):
    """Settlement transaction status"""
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    NETTING = "NETTING"
    SETTLING = "SETTLING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REVERSED = "REVERSED"


class PaymentPriority(Enum):
    """Payment priority levels"""
    CRITICAL = 1  # Central bank, government
    HIGH = 2      # Large value, time-critical
    NORMAL = 3    # Standard payments
    LOW = 4       # Batch, non-urgent


class LiquidityStatus(Enum):
    """Account liquidity status"""
    SUFFICIENT = "SUFFICIENT"
    MARGINAL = "MARGINAL"
    INSUFFICIENT = "INSUFFICIENT"
    FROZEN = "FROZEN"


@dataclass
class SettlementAccount:
    """Settlement account with real-time position tracking"""
    account_id: str
    institution_id: str
    currency: str
    balance: Decimal = Decimal('0')
    available_balance: Decimal = Decimal('0')
    reserved_balance: Decimal = Decimal('0')
    credit_limit: Decimal = Decimal('0')
    
    # Limits and controls
    daily_debit_limit: Decimal = Decimal('0')
    daily_credit_limit: Decimal = Decimal('0')
    daily_debits: Decimal = Decimal('0')
    daily_credits: Decimal = Decimal('0')
    
    # Nostro/Vostro for correspondent banking
    nostro_accounts: Dict[str, str] = field(default_factory=dict)
    vostro_accounts: Dict[str, str] = field(default_factory=dict)
    
    # Real-time metrics
    pending_debits: Decimal = Decimal('0')
    pending_credits: Decimal = Decimal('0')
    last_activity: float = field(default_factory=time.time)
    
    def get_available_liquidity(self) -> Decimal:
        """Calculate available liquidity"""
        return self.available_balance + self.credit_limit - self.pending_debits
    
    def can_debit(self, amount: Decimal) -> bool:
        """Check if debit is possible"""
        if self.daily_debits + amount > self.daily_debit_limit:
            return False
        return self.get_available_liquidity() >= amount
    
    def reserve_funds(self, amount: Decimal):
        """Reserve funds for pending settlement"""
        self.available_balance -= amount
        self.reserved_balance += amount
        self.pending_debits += amount
    
    def release_reservation(self, amount: Decimal):
        """Release reserved funds"""
        self.reserved_balance -= amount
        self.available_balance += amount
        self.pending_debits -= amount
    
    def commit_debit(self, amount: Decimal):
        """Commit debit transaction"""
        self.reserved_balance -= amount
        self.balance -= amount
        self.daily_debits += amount
        self.pending_debits -= amount
        self.last_activity = time.time()
    
    def commit_credit(self, amount: Decimal):
        """Commit credit transaction"""
        self.balance += amount
        self.available_balance += amount
        self.daily_credits += amount
        self.pending_credits -= amount
        self.last_activity = time.time()


@dataclass
class SettlementInstruction:
    """Settlement instruction with full tracking"""
    instruction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    transaction_ref: str = ""
    
    # Parties
    debtor_account: str = ""
    creditor_account: str = ""
    debtor_institution: str = ""
    creditor_institution: str = ""
    
    # Amount and currency
    amount: Decimal = Decimal('0')
    currency: str = "USD"
    
    # Cross-currency
    settlement_currency: str = ""
    exchange_rate: Optional[Decimal] = None
    settlement_amount: Optional[Decimal] = None
    
    # Timing
    value_date: datetime = field(default_factory=datetime.now)
    settlement_date: Optional[datetime] = None
    created_at: float = field(default_factory=time.time)
    settled_at: Optional[float] = None
    
    # Status
    status: SettlementStatus = SettlementStatus.PENDING
    priority: PaymentPriority = PaymentPriority.NORMAL
    
    # Netting
    netting_eligible: bool = True
    netting_group: Optional[str] = None
    netted_amount: Optional[Decimal] = None
    
    # Atomic operations
    atomic_group: Optional[str] = None
    atomic_dependencies: List[str] = field(default_factory=list)
    
    # Compliance
    aml_checked: bool = False
    sanctions_checked: bool = False
    regulatory_reporting: Dict[str, Any] = field(default_factory=dict)
    
    # Error handling
    failure_reason: Optional[str] = None
    retry_count: int = 0
    reversible: bool = True


class LiquidityManager:
    """Real-time liquidity management"""
    
    def __init__(self):
        self.accounts: Dict[str, SettlementAccount] = {}
        self.liquidity_pools: Dict[str, Decimal] = {}
        self.credit_lines: Dict[Tuple[str, str], Decimal] = {}
        self.lock = asyncio.Lock()
    
    async def check_liquidity(self, instruction: SettlementInstruction) -> LiquidityStatus:
        """Check liquidity for settlement"""
        async with self.lock:
            debtor = self.accounts.get(instruction.debtor_account)
            
            if not debtor:
                return LiquidityStatus.INSUFFICIENT
            
            if debtor.balance < Decimal('0'):
                return LiquidityStatus.FROZEN
            
            available = debtor.get_available_liquidity()
            
            if available >= instruction.amount:
                return LiquidityStatus.SUFFICIENT
            elif available >= instruction.amount * (1 - LIQUIDITY_BUFFER_RATIO):
                return LiquidityStatus.MARGINAL
            else:
                return LiquidityStatus.INSUFFICIENT
    
    async def reserve_liquidity(self, instruction: SettlementInstruction) -> bool:
        """Reserve liquidity for settlement"""
        async with self.lock:
            debtor = self.accounts.get(instruction.debtor_account)
            
            if not debtor or not debtor.can_debit(instruction.amount):
                return False
            
            debtor.reserve_funds(instruction.amount)
            
            # Pre-credit for atomic operations
            if instruction.atomic_group:
                creditor = self.accounts.get(instruction.creditor_account)
                if creditor:
                    creditor.pending_credits += instruction.amount
            
            return True
    
    async def release_liquidity(self, instruction: SettlementInstruction):
        """Release reserved liquidity"""
        async with self.lock:
            debtor = self.accounts.get(instruction.debtor_account)
            if debtor:
                debtor.release_reservation(instruction.amount)
            
            if instruction.atomic_group:
                creditor = self.accounts.get(instruction.creditor_account)
                if creditor:
                    creditor.pending_credits -= instruction.amount
    
    async def get_liquidity_forecast(self, account_id: str, 
                                    horizon_minutes: int = 60) -> Dict[str, Decimal]:
        """Forecast liquidity based on pending settlements"""
        async with self.lock:
            account = self.accounts.get(account_id)
            if not account:
                return {}
            
            return {
                'current': account.balance,
                'available': account.get_available_liquidity(),
                'pending_debits': account.pending_debits,
                'pending_credits': account.pending_credits,
                'forecast_balance': account.balance - account.pending_debits + account.pending_credits,
                'credit_available': account.credit_limit - account.pending_debits
            }


class NettingEngine:
    """Multilateral netting engine"""
    
    def __init__(self):
        self.netting_sets: Dict[str, List[SettlementInstruction]] = {}
        self.netting_matrix: Dict[Tuple[str, str], Decimal] = {}
        self.lock = asyncio.Lock()
    
    async def add_to_netting(self, instruction: SettlementInstruction):
        """Add instruction to netting set"""
        if not instruction.netting_eligible:
            return
        
        async with self.lock:
            # Group by currency and settlement date
            group_key = f"{instruction.currency}_{instruction.value_date.date()}"
            
            if group_key not in self.netting_sets:
                self.netting_sets[group_key] = []
            
            self.netting_sets[group_key].append(instruction)
            instruction.netting_group = group_key
    
    async def calculate_net_positions(self, group_key: str) -> Dict[str, Decimal]:
        """Calculate net positions for netting group"""
        async with self.lock:
            if group_key not in self.netting_sets:
                return {}
            
            positions = {}
            
            for instruction in self.netting_sets[group_key]:
                # Calculate net position for each participant
                if instruction.debtor_institution not in positions:
                    positions[instruction.debtor_institution] = Decimal('0')
                if instruction.creditor_institution not in positions:
                    positions[instruction.creditor_institution] = Decimal('0')
                
                positions[instruction.debtor_institution] -= instruction.amount
                positions[instruction.creditor_institution] += instruction.amount
            
            return positions
    
    async def generate_netted_instructions(self, group_key: str) -> List[SettlementInstruction]:
        """Generate netted settlement instructions"""
        positions = await self.calculate_net_positions(group_key)
        
        netted_instructions = []
        
        # Create bilateral net settlements
        debtors = {k: v for k, v in positions.items() if v < 0}
        creditors = {k: v for k, v in positions.items() if v > 0}
        
        for debtor, debit_amount in debtors.items():
            for creditor, credit_amount in creditors.items():
                if debit_amount == 0 or credit_amount == 0:
                    continue
                
                amount = min(abs(debit_amount), credit_amount)
                
                netted = SettlementInstruction(
                    debtor_institution=debtor,
                    creditor_institution=creditor,
                    amount=amount,
                    currency=group_key.split('_')[0],
                    netting_group=group_key,
                    priority=PaymentPriority.HIGH
                )
                
                netted_instructions.append(netted)
                
                debit_amount += amount
                credit_amount -= amount
        
        return netted_instructions


class AtomicSettlement:
    """Atomic settlement coordinator for multi-leg transactions"""
    
    def __init__(self, settlement_engine: 'RealtimeSettlementEngine'):
        self.engine = settlement_engine
        self.atomic_groups: Dict[str, List[SettlementInstruction]] = {}
        self.group_status: Dict[str, str] = {}
        self.lock = asyncio.Lock()
    
    async def create_atomic_group(self, instructions: List[SettlementInstruction]) -> str:
        """Create atomic settlement group"""
        group_id = str(uuid.uuid4())
        
        async with self.lock:
            # Set atomic group for all instructions
            for instruction in instructions:
                instruction.atomic_group = group_id
                instruction.atomic_dependencies = [
                    i.instruction_id for i in instructions 
                    if i.instruction_id != instruction.instruction_id
                ]
            
            self.atomic_groups[group_id] = instructions
            self.group_status[group_id] = "PREPARING"
        
        return group_id
    
    async def validate_atomic_group(self, group_id: str) -> bool:
        """Validate all legs of atomic settlement"""
        async with self.lock:
            if group_id not in self.atomic_groups:
                return False
            
            instructions = self.atomic_groups[group_id]
            
            # Check liquidity for all legs
            for instruction in instructions:
                status = await self.engine.liquidity_manager.check_liquidity(instruction)
                if status != LiquidityStatus.SUFFICIENT:
                    logger.warning(f"Insufficient liquidity for atomic group {group_id}")
                    return False
            
            # Reserve liquidity for all legs
            for instruction in instructions:
                if not await self.engine.liquidity_manager.reserve_liquidity(instruction):
                    # Rollback reservations
                    await self.rollback_atomic_group(group_id)
                    return False
            
            self.group_status[group_id] = "VALIDATED"
            return True
    
    async def execute_atomic_group(self, group_id: str) -> bool:
        """Execute atomic settlement group"""
        async with self.lock:
            if group_id not in self.atomic_groups:
                return False
            
            if self.group_status[group_id] != "VALIDATED":
                return False
            
            instructions = self.atomic_groups[group_id]
            
            try:
                # Execute all legs
                for instruction in instructions:
                    instruction.status = SettlementStatus.SETTLING
                
                # Simulate settlement (would integrate with actual ledger)
                await asyncio.sleep(0.01)
                
                # Commit all legs
                for instruction in instructions:
                    instruction.status = SettlementStatus.COMPLETED
                    instruction.settled_at = time.time()
                
                self.group_status[group_id] = "COMPLETED"
                return True
                
            except Exception as e:
                logger.error(f"Atomic settlement failed: {e}")
                await self.rollback_atomic_group(group_id)
                return False
    
    async def rollback_atomic_group(self, group_id: str):
        """Rollback atomic settlement group"""
        async with self.lock:
            if group_id not in self.atomic_groups:
                return
            
            instructions = self.atomic_groups[group_id]
            
            for instruction in instructions:
                # Release reserved liquidity
                await self.engine.liquidity_manager.release_liquidity(instruction)
                
                # Update status
                instruction.status = SettlementStatus.FAILED
                instruction.failure_reason = "Atomic group rollback"
            
            self.group_status[group_id] = "ROLLED_BACK"


class CrossBorderSettlement:
    """Cross-border payment settlement with correspondent banking"""
    
    def __init__(self):
        self.correspondent_banks: Dict[str, Dict[str, Any]] = {}
        self.fx_rates: Dict[Tuple[str, str], Decimal] = {}
        self.nostro_positions: Dict[str, Decimal] = {}
        self.vostro_positions: Dict[str, Decimal] = {}
        self.pending_confirmations: Dict[str, SettlementInstruction] = {}
    
    async def route_cross_border(self, instruction: SettlementInstruction) -> Optional[str]:
        """Route cross-border payment through correspondent network"""
        
        # Find correspondent bank route
        route = await self._find_correspondent_route(
            instruction.debtor_institution,
            instruction.creditor_institution,
            instruction.currency
        )
        
        if not route:
            return None
        
        # Convert currency if needed
        if instruction.currency != instruction.settlement_currency:
            rate = await self.get_fx_rate(instruction.currency, instruction.settlement_currency)
            instruction.exchange_rate = rate
            instruction.settlement_amount = instruction.amount * rate
        else:
            instruction.settlement_amount = instruction.amount
        
        # Update nostro/vostro positions
        await self._update_correspondent_positions(instruction, route)
        
        # Send settlement message (MT202/MT103)
        message_id = await self._send_swift_message(instruction, route)
        
        return message_id
    
    async def _find_correspondent_route(self, debtor: str, creditor: str, 
                                       currency: str) -> Optional[List[str]]:
        """Find optimal correspondent banking route"""
        # Simplified routing - in production would use graph algorithms
        
        if currency == "USD":
            return ["JPMC", "CORRESPONDENT_BANK", creditor]
        elif currency == "EUR":
            return ["DEUTSCHE_BANK", "CORRESPONDENT_BANK", creditor]
        elif currency == "GBP":
            return ["BARCLAYS", "CORRESPONDENT_BANK", creditor]
        else:
            # Find route through correspondent network
            return None
    
    async def get_fx_rate(self, from_currency: str, to_currency: str) -> Decimal:
        """Get real-time FX rate"""
        key = (from_currency, to_currency)
        
        # Check cache
        if key in self.fx_rates:
            return self.fx_rates[key]
        
        # Fetch from market data (simplified)
        rates = {
            ("USD", "EUR"): Decimal("0.92"),
            ("EUR", "USD"): Decimal("1.09"),
            ("USD", "GBP"): Decimal("0.79"),
            ("GBP", "USD"): Decimal("1.27"),
            ("EUR", "GBP"): Decimal("0.86"),
            ("GBP", "EUR"): Decimal("1.16")
        }
        
        rate = rates.get(key, Decimal("1"))
        self.fx_rates[key] = rate
        
        return rate
    
    async def _update_correspondent_positions(self, instruction: SettlementInstruction,
                                             route: List[str]):
        """Update nostro/vostro positions"""
        # Debit nostro account
        nostro_key = f"{instruction.debtor_institution}_{route[0]}_{instruction.currency}"
        if nostro_key not in self.nostro_positions:
            self.nostro_positions[nostro_key] = Decimal("0")
        
        self.nostro_positions[nostro_key] -= instruction.amount
        
        # Credit vostro account at correspondent
        vostro_key = f"{route[0]}_{instruction.creditor_institution}_{instruction.currency}"
        if vostro_key not in self.vostro_positions:
            self.vostro_positions[vostro_key] = Decimal("0")
        
        self.vostro_positions[vostro_key] += instruction.amount
    
    async def _send_swift_message(self, instruction: SettlementInstruction,
                                 route: List[str]) -> str:
        """Send SWIFT MT202/MT103 message"""
        message_id = f"SWIFT_{uuid.uuid4().hex[:16].upper()}"
        
        # In production, would actually send SWIFT message
        logger.info(f"Sending SWIFT message {message_id} via route {route}")
        
        # Store for confirmation tracking
        self.pending_confirmations[message_id] = instruction
        
        return message_id
    
    async def process_swift_confirmation(self, message_id: str, status: str):
        """Process SWIFT confirmation message (MT900/MT910)"""
        if message_id not in self.pending_confirmations:
            logger.warning(f"Unknown SWIFT confirmation: {message_id}")
            return
        
        instruction = self.pending_confirmations[message_id]
        
        if status == "CONFIRMED":
            instruction.status = SettlementStatus.COMPLETED
            instruction.settled_at = time.time()
        else:
            instruction.status = SettlementStatus.FAILED
            instruction.failure_reason = f"SWIFT confirmation failed: {status}"
        
        del self.pending_confirmations[message_id]


class RealtimeSettlementEngine:
    """Main real-time gross settlement engine"""
    
    def __init__(self):
        self.liquidity_manager = LiquidityManager()
        self.netting_engine = NettingEngine()
        self.atomic_coordinator = AtomicSettlement(self)
        self.cross_border = CrossBorderSettlement()
        
        self.instruction_queue = PriorityQueue()
        self.processing_queue = asyncio.Queue()
        self.completed_settlements: List[SettlementInstruction] = []
        
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=50)
        
        # Persistence layer (Redis for speed)
        self.redis_client = None  # Would connect to Redis
        
        # Metrics
        self.metrics = {
            'total_settled': 0,
            'total_value': Decimal('0'),
            'average_latency_ms': 0,
            'success_rate': 100.0,
            'netting_efficiency': 0.0
        }
    
    async def start(self):
        """Start settlement engine"""
        self.running = True
        
        # Start processing workers
        asyncio.create_task(self._instruction_processor())
        asyncio.create_task(self._settlement_worker())
        asyncio.create_task(self._netting_processor())
        asyncio.create_task(self._metric_collector())
        
        logger.info("Real-time settlement engine started")
    
    async def submit_instruction(self, instruction: SettlementInstruction) -> str:
        """Submit settlement instruction"""
        
        # Validate instruction
        if not await self._validate_instruction(instruction):
            instruction.status = SettlementStatus.FAILED
            instruction.failure_reason = "Validation failed"
            return instruction.instruction_id
        
        # Add to processing queue
        priority = instruction.priority.value
        self.instruction_queue.put((priority, time.time(), instruction))
        
        logger.info(f"Settlement instruction {instruction.instruction_id} submitted")
        
        return instruction.instruction_id
    
    async def _validate_instruction(self, instruction: SettlementInstruction) -> bool:
        """Validate settlement instruction"""
        
        # Amount validation
        if instruction.amount <= 0:
            return False
        
        if instruction.amount > MAX_SINGLE_TRANSACTION:
            logger.warning(f"Transaction exceeds maximum: {instruction.amount}")
            return False
        
        # Account validation
        if not instruction.debtor_account or not instruction.creditor_account:
            return False
        
        # Compliance checks
        if not instruction.aml_checked:
            # Perform AML check
            instruction.aml_checked = await self._perform_aml_check(instruction)
            if not instruction.aml_checked:
                return False
        
        if not instruction.sanctions_checked:
            # Perform sanctions check
            instruction.sanctions_checked = await self._perform_sanctions_check(instruction)
            if not instruction.sanctions_checked:
                return False
        
        return True
    
    async def _perform_aml_check(self, instruction: SettlementInstruction) -> bool:
        """Perform AML compliance check"""
        # Simplified AML check - would integrate with actual AML system
        
        # Check transaction patterns
        suspicious_patterns = [
            instruction.amount == round(instruction.amount / 1000) * 1000,  # Round amounts
            instruction.amount == Decimal("9999"),  # Just below reporting threshold
        ]
        
        if any(suspicious_patterns):
            instruction.regulatory_reporting['aml_flag'] = "SUSPICIOUS"
            # Would file SAR in production
            return False
        
        return True
    
    async def _perform_sanctions_check(self, instruction: SettlementInstruction) -> bool:
        """Perform sanctions screening"""
        # Would integrate with actual sanctions lists (OFAC, EU, UN)
        sanctioned_entities = []  # Would load from sanctions database
        
        for entity in sanctioned_entities:
            if entity in [instruction.debtor_institution, instruction.creditor_institution]:
                return False
        
        return True
    
    async def _instruction_processor(self):
        """Process settlement instructions"""
        while self.running:
            try:
                if not self.instruction_queue.empty():
                    priority, timestamp, instruction = self.instruction_queue.get_nowait()
                    
                    # Check for netting eligibility
                    if instruction.netting_eligible and not instruction.atomic_group:
                        await self.netting_engine.add_to_netting(instruction)
                    else:
                        await self.processing_queue.put(instruction)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Instruction processor error: {e}")
    
    async def _settlement_worker(self):
        """Execute settlements"""
        while self.running:
            try:
                instruction = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=0.1
                )
                
                start_time = time.time()
                
                # Execute settlement
                success = await self._execute_settlement(instruction)
                
                # Update metrics
                latency_ms = (time.time() - start_time) * 1000
                self._update_metrics(instruction, success, latency_ms)
                
                # Store completed settlement
                if success:
                    self.completed_settlements.append(instruction)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Settlement worker error: {e}")
    
    async def _execute_settlement(self, instruction: SettlementInstruction) -> bool:
        """Execute single settlement"""
        
        instruction.status = SettlementStatus.SETTLING
        
        try:
            # Check liquidity
            liquidity_status = await self.liquidity_manager.check_liquidity(instruction)
            
            if liquidity_status == LiquidityStatus.INSUFFICIENT:
                # Try liquidity optimization
                if not await self._optimize_liquidity(instruction):
                    instruction.status = SettlementStatus.FAILED
                    instruction.failure_reason = "Insufficient liquidity"
                    return False
            
            # Reserve funds
            if not await self.liquidity_manager.reserve_liquidity(instruction):
                instruction.status = SettlementStatus.FAILED
                instruction.failure_reason = "Failed to reserve liquidity"
                return False
            
            # Execute based on type
            if instruction.atomic_group:
                # Part of atomic settlement
                success = await self.atomic_coordinator.execute_atomic_group(
                    instruction.atomic_group
                )
            elif instruction.settlement_currency and \
                 instruction.settlement_currency != instruction.currency:
                # Cross-border settlement
                message_id = await self.cross_border.route_cross_border(instruction)
                success = message_id is not None
            else:
                # Standard settlement
                success = await self._perform_ledger_update(instruction)
            
            if success:
                instruction.status = SettlementStatus.COMPLETED
                instruction.settled_at = time.time()
            else:
                instruction.status = SettlementStatus.FAILED
                await self.liquidity_manager.release_liquidity(instruction)
            
            return success
            
        except Exception as e:
            logger.error(f"Settlement execution error: {e}")
            instruction.status = SettlementStatus.FAILED
            instruction.failure_reason = str(e)
            return False
    
    async def _perform_ledger_update(self, instruction: SettlementInstruction) -> bool:
        """Update ledger for settlement"""
        
        # Get accounts
        debtor = self.liquidity_manager.accounts.get(instruction.debtor_account)
        creditor = self.liquidity_manager.accounts.get(instruction.creditor_account)
        
        if not debtor or not creditor:
            return False
        
        # Perform double-entry bookkeeping
        async with self.liquidity_manager.lock:
            debtor.commit_debit(instruction.amount)
            creditor.commit_credit(instruction.amount)
        
        # Record in immutable ledger (would use blockchain/DLT)
        await self._record_to_ledger(instruction)
        
        return True
    
    async def _record_to_ledger(self, instruction: SettlementInstruction):
        """Record settlement to immutable ledger"""
        # Would integrate with blockchain/DLT
        
        ledger_entry = {
            'instruction_id': instruction.instruction_id,
            'timestamp': instruction.settled_at,
            'debtor': instruction.debtor_account,
            'creditor': instruction.creditor_account,
            'amount': str(instruction.amount),
            'currency': instruction.currency,
            'hash': hashlib.sha256(
                f"{instruction.instruction_id}{instruction.settled_at}".encode()
            ).hexdigest()
        }
        
        # Store in distributed ledger
        logger.debug(f"Ledger entry: {ledger_entry}")
    
    async def _optimize_liquidity(self, instruction: SettlementInstruction) -> bool:
        """Optimize liquidity through various mechanisms"""
        
        # Try netting first
        if instruction.netting_eligible:
            netted = await self.netting_engine.generate_netted_instructions(
                instruction.netting_group
            )
            if netted:
                # Replace with netted instructions
                for netted_instruction in netted:
                    await self.processing_queue.put(netted_instruction)
                return True
        
        # Try liquidity pooling
        # Would implement liquidity sharing mechanisms
        
        # Try intraday credit
        # Would check credit lines and collateral
        
        return False
    
    async def _netting_processor(self):
        """Process netting cycles"""
        while self.running:
            try:
                await asyncio.sleep(NETTING_WINDOW_MS / 1000)
                
                # Process each netting group
                for group_key in list(self.netting_engine.netting_sets.keys()):
                    netted = await self.netting_engine.generate_netted_instructions(group_key)
                    
                    for instruction in netted:
                        await self.processing_queue.put(instruction)
                    
                    # Clear processed group
                    del self.netting_engine.netting_sets[group_key]
                
            except Exception as e:
                logger.error(f"Netting processor error: {e}")
    
    async def _metric_collector(self):
        """Collect and report metrics"""
        while self.running:
            await asyncio.sleep(60)  # Report every minute
            
            logger.info(f"Settlement metrics: {self.metrics}")
    
    def _update_metrics(self, instruction: SettlementInstruction, 
                       success: bool, latency_ms: float):
        """Update performance metrics"""
        self.metrics['total_settled'] += 1
        
        if success:
            self.metrics['total_value'] += instruction.amount
        
        # Update average latency
        current_avg = self.metrics['average_latency_ms']
        total = self.metrics['total_settled']
        self.metrics['average_latency_ms'] = (
            (current_avg * (total - 1) + latency_ms) / total
        )
        
        # Update success rate
        if not success:
            self.metrics['success_rate'] = (
                (self.metrics['success_rate'] * (total - 1)) / total
            ) * 100
    
    async def get_settlement_status(self, instruction_id: str) -> Optional[SettlementInstruction]:
        """Get status of settlement instruction"""
        # Check completed settlements
        for instruction in self.completed_settlements:
            if instruction.instruction_id == instruction_id:
                return instruction
        
        # Check pending queues
        # Would search through active queues
        
        return None
    
    async def stop(self):
        """Stop settlement engine"""
        self.running = False
        self.executor.shutdown(wait=True)
        logger.info("Settlement engine stopped")


async def main():
    """Test real-time settlement engine"""
    engine = RealtimeSettlementEngine()
    
    # Create test accounts
    engine.liquidity_manager.accounts = {
        "ACC001": SettlementAccount(
            account_id="ACC001",
            institution_id="BANK001",
            currency="USD",
            balance=Decimal("1000000"),
            available_balance=Decimal("1000000"),
            daily_debit_limit=Decimal("5000000"),
            daily_credit_limit=Decimal("10000000")
        ),
        "ACC002": SettlementAccount(
            account_id="ACC002",
            institution_id="BANK002",
            currency="USD",
            balance=Decimal("500000"),
            available_balance=Decimal("500000"),
            daily_debit_limit=Decimal("5000000"),
            daily_credit_limit=Decimal("10000000")
        )
    }
    
    await engine.start()
    
    # Test single settlement
    instruction1 = SettlementInstruction(
        debtor_account="ACC001",
        creditor_account="ACC002",
        debtor_institution="BANK001",
        creditor_institution="BANK002",
        amount=Decimal("10000"),
        currency="USD",
        priority=PaymentPriority.HIGH,
        aml_checked=True,
        sanctions_checked=True
    )
    
    await engine.submit_instruction(instruction1)
    
    # Test atomic settlement
    atomic_instructions = [
        SettlementInstruction(
            debtor_account="ACC001",
            creditor_account="ACC002",
            amount=Decimal("5000"),
            currency="USD"
        ),
        SettlementInstruction(
            debtor_account="ACC002",
            creditor_account="ACC001",
            amount=Decimal("3000"),
            currency="USD"
        )
    ]
    
    group_id = await engine.atomic_coordinator.create_atomic_group(atomic_instructions)
    
    if await engine.atomic_coordinator.validate_atomic_group(group_id):
        await engine.atomic_coordinator.execute_atomic_group(group_id)
    
    # Wait for settlements
    await asyncio.sleep(2)
    
    # Check results
    print(f"Completed settlements: {len(engine.completed_settlements)}")
    print(f"Metrics: {engine.metrics}")
    
    # Check account balances
    for acc_id, account in engine.liquidity_manager.accounts.items():
        print(f"Account {acc_id}: Balance={account.balance}, Available={account.available_balance}")
    
    await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())