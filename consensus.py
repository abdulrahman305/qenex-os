#!/usr/bin/env python3
"""
QENEX Consensus Mechanism
Byzantine Fault Tolerant consensus with Proof of Stake
"""

import hashlib
import time
import asyncio
import random
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
import json
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, utils
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

# Configuration
BLOCK_TIME = 10  # seconds
COMMITTEE_SIZE = 21
MINIMUM_STAKE = Decimal('10000')
SLASH_RATE = Decimal('0.1')  # 10%
FINALITY_THRESHOLD = 0.67  # 2/3 + 1
MAX_ROUNDS = 10
PROPOSAL_TIMEOUT = 5  # seconds
VOTE_TIMEOUT = 5  # seconds

class ConsensusPhase(Enum):
    """Consensus phases"""
    IDLE = "idle"
    PROPOSING = "proposing"
    PREVOTING = "prevoting"
    PRECOMMITTING = "precommitting"
    COMMITTING = "committing"
    FINALIZED = "finalized"

class VoteType(Enum):
    """Vote types"""
    PREVOTE = "prevote"
    PRECOMMIT = "precommit"
    COMMIT = "commit"

@dataclass
class Validator:
    """Validator information"""
    address: str
    stake: Decimal
    public_key: Any
    reputation: int = 100
    last_block_proposed: int = 0
    last_vote_time: float = 0
    slashed: bool = False
    
    def voting_power(self) -> Decimal:
        """Calculate voting power based on stake and reputation"""
        if self.slashed:
            return Decimal('0')
        return self.stake * Decimal(self.reputation) / Decimal('100')

@dataclass
class Vote:
    """Consensus vote"""
    validator: str
    block_hash: str
    vote_type: VoteType
    round: int
    timestamp: float
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'validator': self.validator,
            'block_hash': self.block_hash,
            'vote_type': self.vote_type.value,
            'round': self.round,
            'timestamp': self.timestamp,
            'signature': self.signature
        }
    
    def sign(self, private_key: Any):
        """Sign the vote"""
        message = f"{self.validator}{self.block_hash}{self.vote_type.value}{self.round}{self.timestamp}"
        message_bytes = message.encode()
        signature = private_key.sign(
            message_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        self.signature = signature.hex()
    
    def verify(self, public_key: Any) -> bool:
        """Verify vote signature"""
        if not self.signature:
            return False
        
        message = f"{self.validator}{self.block_hash}{self.vote_type.value}{self.round}{self.timestamp}"
        message_bytes = message.encode()
        
        try:
            public_key.verify(
                bytes.fromhex(self.signature),
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except InvalidSignature:
            return False

@dataclass
class ConsensusState:
    """Current consensus state"""
    height: int
    round: int
    phase: ConsensusPhase
    proposed_block: Optional[Any] = None
    proposer: Optional[str] = None
    prevotes: Dict[str, Vote] = field(default_factory=dict)
    precommits: Dict[str, Vote] = field(default_factory=dict)
    commits: Dict[str, Vote] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    locked_block: Optional[Any] = None
    locked_round: int = -1

class BFTConsensus:
    """Byzantine Fault Tolerant Consensus Implementation"""
    
    def __init__(self, node_id: str, blockchain: Any, network: Any):
        self.node_id = node_id
        self.blockchain = blockchain
        self.network = network
        
        # Validator set
        self.validators: Dict[str, Validator] = {}
        self.active_validators: List[str] = []
        
        # Consensus state
        self.state = ConsensusState(
            height=0,
            round=0,
            phase=ConsensusPhase.IDLE
        )
        
        # Own validator info
        self.private_key = None
        self.public_key = None
        self.is_validator = False
        
        # Consensus parameters
        self.block_time = BLOCK_TIME
        self.timeout_propose = PROPOSAL_TIMEOUT
        self.timeout_prevote = VOTE_TIMEOUT
        self.timeout_precommit = VOTE_TIMEOUT
        
        # Event handlers
        self.on_new_block = None
        self.on_finalized = None
        
        # Statistics
        self.blocks_proposed = 0
        self.blocks_finalized = 0
        self.consensus_failures = 0
    
    def initialize_validator(self, private_key: Any, stake: Decimal):
        """Initialize as a validator"""
        self.private_key = private_key
        self.public_key = private_key.public_key()
        
        validator = Validator(
            address=self.node_id,
            stake=stake,
            public_key=self.public_key
        )
        
        self.validators[self.node_id] = validator
        self.is_validator = True
    
    def add_validator(self, address: str, stake: Decimal, public_key: Any):
        """Add a validator to the set"""
        if stake >= MINIMUM_STAKE:
            validator = Validator(
                address=address,
                stake=stake,
                public_key=public_key
            )
            self.validators[address] = validator
    
    def update_active_validators(self):
        """Update active validator set based on stake"""
        # Sort validators by voting power
        sorted_validators = sorted(
            self.validators.items(),
            key=lambda x: x[1].voting_power(),
            reverse=True
        )
        
        # Select top validators
        self.active_validators = [
            addr for addr, _ in sorted_validators[:COMMITTEE_SIZE]
            if not self.validators[addr].slashed
        ]
    
    def get_proposer(self, height: int, round: int) -> str:
        """Deterministically select block proposer"""
        if not self.active_validators:
            return None
        
        # Use height and round for deterministic selection
        seed = hashlib.sha256(f"{height}{round}".encode()).hexdigest()
        index = int(seed, 16) % len(self.active_validators)
        
        return self.active_validators[index]
    
    async def start_consensus(self):
        """Start consensus process"""
        while True:
            try:
                # Update validator set
                self.update_active_validators()
                
                # Get current height
                current_height = len(self.blockchain.chain)
                
                if self.state.height < current_height:
                    # Start new consensus round
                    self.state = ConsensusState(
                        height=current_height,
                        round=0,
                        phase=ConsensusPhase.PROPOSING
                    )
                    
                    await self.run_consensus_round()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Consensus error: {e}")
                self.consensus_failures += 1
                await asyncio.sleep(5)
    
    async def run_consensus_round(self):
        """Run a single consensus round"""
        proposer = self.get_proposer(self.state.height, self.state.round)
        
        if not proposer:
            return
        
        self.state.proposer = proposer
        
        # Phase 1: Proposal
        if proposer == self.node_id:
            await self.propose_block()
        else:
            await self.wait_for_proposal()
        
        # Phase 2: Prevote
        self.state.phase = ConsensusPhase.PREVOTING
        await self.prevote()
        await self.wait_for_prevotes()
        
        # Phase 3: Precommit
        self.state.phase = ConsensusPhase.PRECOMMITTING
        await self.precommit()
        await self.wait_for_precommits()
        
        # Phase 4: Commit
        self.state.phase = ConsensusPhase.COMMITTING
        await self.commit()
        
        # Check if we need another round
        if self.state.phase != ConsensusPhase.FINALIZED:
            if self.state.round < MAX_ROUNDS:
                self.state.round += 1
                await self.run_consensus_round()
            else:
                # Consensus failed after max rounds
                self.consensus_failures += 1
                self.state.phase = ConsensusPhase.IDLE
    
    async def propose_block(self):
        """Propose a new block"""
        if not self.is_validator:
            return
        
        # Create new block
        transactions = self.blockchain.mempool.get_transactions()
        
        block = self.blockchain.create_block(
            transactions=transactions,
            proposer=self.node_id
        )
        
        self.state.proposed_block = block
        self.blocks_proposed += 1
        
        # Broadcast proposal
        proposal_msg = {
            'type': 'consensus_proposal',
            'height': self.state.height,
            'round': self.state.round,
            'block': block.to_dict(),
            'proposer': self.node_id,
            'timestamp': time.time()
        }
        
        await self.network.broadcast(proposal_msg)
    
    async def wait_for_proposal(self):
        """Wait for block proposal"""
        timeout = time.time() + self.timeout_propose
        
        while time.time() < timeout:
            # Check if proposal received
            if self.state.proposed_block:
                return
            
            await asyncio.sleep(0.1)
        
        # Timeout - no proposal received
        self.state.proposed_block = None
    
    async def prevote(self):
        """Send prevote for proposed block"""
        if not self.is_validator or self.node_id not in self.active_validators:
            return
        
        # Decide what to vote for
        vote_hash = None
        
        if self.state.proposed_block and self.validate_block(self.state.proposed_block):
            vote_hash = self.state.proposed_block.block_hash
        elif self.state.locked_block:
            vote_hash = self.state.locked_block.block_hash
        
        if vote_hash:
            vote = Vote(
                validator=self.node_id,
                block_hash=vote_hash,
                vote_type=VoteType.PREVOTE,
                round=self.state.round,
                timestamp=time.time()
            )
            
            vote.sign(self.private_key)
            self.state.prevotes[self.node_id] = vote
            
            # Broadcast vote
            await self.broadcast_vote(vote)
    
    async def wait_for_prevotes(self):
        """Wait for prevotes from validators"""
        timeout = time.time() + self.timeout_prevote
        
        while time.time() < timeout:
            # Check if we have enough prevotes
            if self.count_votes(self.state.prevotes) >= self.get_threshold():
                return
            
            await asyncio.sleep(0.1)
    
    async def precommit(self):
        """Send precommit based on prevotes"""
        if not self.is_validator or self.node_id not in self.active_validators:
            return
        
        # Check prevote results
        prevote_block = self.get_majority_block(self.state.prevotes)
        
        if prevote_block:
            # Lock on this block
            self.state.locked_block = self.state.proposed_block
            self.state.locked_round = self.state.round
            
            vote = Vote(
                validator=self.node_id,
                block_hash=prevote_block,
                vote_type=VoteType.PRECOMMIT,
                round=self.state.round,
                timestamp=time.time()
            )
            
            vote.sign(self.private_key)
            self.state.precommits[self.node_id] = vote
            
            await self.broadcast_vote(vote)
    
    async def wait_for_precommits(self):
        """Wait for precommits from validators"""
        timeout = time.time() + self.timeout_precommit
        
        while time.time() < timeout:
            # Check if we have enough precommits
            if self.count_votes(self.state.precommits) >= self.get_threshold():
                return
            
            await asyncio.sleep(0.1)
    
    async def commit(self):
        """Commit block if consensus reached"""
        precommit_block = self.get_majority_block(self.state.precommits)
        
        if precommit_block and self.state.proposed_block:
            if precommit_block == self.state.proposed_block.block_hash:
                # Consensus reached - finalize block
                self.blockchain.add_block(self.state.proposed_block)
                self.state.phase = ConsensusPhase.FINALIZED
                self.blocks_finalized += 1
                
                # Send commit message
                vote = Vote(
                    validator=self.node_id,
                    block_hash=precommit_block,
                    vote_type=VoteType.COMMIT,
                    round=self.state.round,
                    timestamp=time.time()
                )
                
                if self.is_validator:
                    vote.sign(self.private_key)
                    await self.broadcast_vote(vote)
                
                # Notify handlers
                if self.on_finalized:
                    await self.on_finalized(self.state.proposed_block)
    
    def validate_block(self, block: Any) -> bool:
        """Validate proposed block"""
        # Check block structure
        if not block or not hasattr(block, 'validate'):
            return False
        
        # Validate block
        if not block.validate():
            return False
        
        # Check proposer
        expected_proposer = self.get_proposer(self.state.height, self.state.round)
        if hasattr(block, 'proposer') and block.proposer != expected_proposer:
            return False
        
        # Validate transactions
        for tx in block.transactions:
            if not tx.verify_signature():
                return False
        
        return True
    
    def count_votes(self, votes: Dict[str, Vote]) -> Decimal:
        """Count voting power of votes"""
        total_power = Decimal('0')
        
        for validator_addr, vote in votes.items():
            if validator_addr in self.validators:
                validator = self.validators[validator_addr]
                if vote.verify(validator.public_key):
                    total_power += validator.voting_power()
        
        return total_power
    
    def get_threshold(self) -> Decimal:
        """Get voting threshold for consensus"""
        total_power = sum(
            self.validators[addr].voting_power()
            for addr in self.active_validators
        )
        return total_power * Decimal(str(FINALITY_THRESHOLD))
    
    def get_majority_block(self, votes: Dict[str, Vote]) -> Optional[str]:
        """Get block hash with majority votes"""
        block_votes: Dict[str, Decimal] = {}
        
        for validator_addr, vote in votes.items():
            if validator_addr in self.validators:
                validator = self.validators[validator_addr]
                if vote.verify(validator.public_key):
                    if vote.block_hash not in block_votes:
                        block_votes[vote.block_hash] = Decimal('0')
                    block_votes[vote.block_hash] += validator.voting_power()
        
        threshold = self.get_threshold()
        
        for block_hash, power in block_votes.items():
            if power >= threshold:
                return block_hash
        
        return None
    
    async def broadcast_vote(self, vote: Vote):
        """Broadcast vote to network"""
        vote_msg = {
            'type': 'consensus_vote',
            'vote': vote.to_dict()
        }
        
        await self.network.broadcast(vote_msg)
    
    async def handle_vote(self, vote_data: Dict):
        """Handle incoming vote"""
        vote = Vote(
            validator=vote_data['validator'],
            block_hash=vote_data['block_hash'],
            vote_type=VoteType[vote_data['vote_type'].upper()],
            round=vote_data['round'],
            timestamp=vote_data['timestamp'],
            signature=vote_data.get('signature')
        )
        
        # Verify vote
        if vote.validator not in self.validators:
            return
        
        validator = self.validators[vote.validator]
        if not vote.verify(validator.public_key):
            return
        
        # Store vote based on type
        if vote.vote_type == VoteType.PREVOTE:
            self.state.prevotes[vote.validator] = vote
        elif vote.vote_type == VoteType.PRECOMMIT:
            self.state.precommits[vote.validator] = vote
        elif vote.vote_type == VoteType.COMMIT:
            self.state.commits[vote.validator] = vote
    
    async def handle_proposal(self, proposal_data: Dict):
        """Handle block proposal"""
        if proposal_data['height'] != self.state.height:
            return
        
        if proposal_data['round'] != self.state.round:
            return
        
        # Verify proposer
        expected_proposer = self.get_proposer(
            proposal_data['height'],
            proposal_data['round']
        )
        
        if proposal_data['proposer'] != expected_proposer:
            return
        
        # Store proposed block
        # In real implementation, reconstruct block from data
        self.state.proposed_block = proposal_data['block']
    
    def slash_validator(self, validator_addr: str, reason: str):
        """Slash a validator for misbehavior"""
        if validator_addr in self.validators:
            validator = self.validators[validator_addr]
            
            # Reduce stake
            slash_amount = validator.stake * SLASH_RATE
            validator.stake -= slash_amount
            
            # Reduce reputation
            validator.reputation = max(0, validator.reputation - 20)
            
            # Mark as slashed if stake too low
            if validator.stake < MINIMUM_STAKE:
                validator.slashed = True
                
                # Remove from active validators
                if validator_addr in self.active_validators:
                    self.active_validators.remove(validator_addr)
            
            print(f"Validator {validator_addr[:8]} slashed: {reason}")
    
    def get_consensus_info(self) -> Dict:
        """Get consensus status information"""
        return {
            'height': self.state.height,
            'round': self.state.round,
            'phase': self.state.phase.value,
            'proposer': self.state.proposer,
            'validators': len(self.active_validators),
            'total_stake': sum(v.stake for v in self.validators.values()),
            'prevotes': len(self.state.prevotes),
            'precommits': len(self.state.precommits),
            'blocks_proposed': self.blocks_proposed,
            'blocks_finalized': self.blocks_finalized,
            'consensus_failures': self.consensus_failures
        }

class ProofOfStake:
    """Proof of Stake implementation"""
    
    def __init__(self, min_stake: Decimal = MINIMUM_STAKE):
        self.min_stake = min_stake
        self.validators: Dict[str, Decimal] = {}
        self.delegations: Dict[str, Dict[str, Decimal]] = {}
        self.rewards_pool = Decimal('0')
        self.total_staked = Decimal('0')
    
    def stake(self, validator: str, amount: Decimal):
        """Stake tokens to become validator"""
        if amount < self.min_stake:
            raise ValueError(f"Minimum stake is {self.min_stake}")
        
        if validator not in self.validators:
            self.validators[validator] = Decimal('0')
        
        self.validators[validator] += amount
        self.total_staked += amount
    
    def unstake(self, validator: str, amount: Decimal):
        """Unstake tokens"""
        if validator not in self.validators:
            raise ValueError("Validator not found")
        
        if self.validators[validator] < amount:
            raise ValueError("Insufficient stake")
        
        self.validators[validator] -= amount
        self.total_staked -= amount
        
        # Remove validator if stake below minimum
        if self.validators[validator] < self.min_stake:
            del self.validators[validator]
    
    def delegate(self, delegator: str, validator: str, amount: Decimal):
        """Delegate stake to validator"""
        if validator not in self.validators:
            raise ValueError("Validator not found")
        
        if delegator not in self.delegations:
            self.delegations[delegator] = {}
        
        if validator not in self.delegations[delegator]:
            self.delegations[delegator][validator] = Decimal('0')
        
        self.delegations[delegator][validator] += amount
        self.total_staked += amount
    
    def calculate_rewards(self, block_reward: Decimal) -> Dict[str, Decimal]:
        """Calculate rewards distribution"""
        rewards = {}
        
        if self.total_staked == 0:
            return rewards
        
        for validator, stake in self.validators.items():
            # Calculate validator's share
            validator_power = stake
            
            # Add delegated stake
            for delegator_stakes in self.delegations.values():
                if validator in delegator_stakes:
                    validator_power += delegator_stakes[validator]
            
            # Calculate reward
            reward_share = validator_power / self.total_staked
            rewards[validator] = block_reward * reward_share
        
        return rewards
    
    def get_validator_set(self, max_validators: int = COMMITTEE_SIZE) -> List[Tuple[str, Decimal]]:
        """Get active validator set"""
        # Sort by stake
        sorted_validators = sorted(
            self.validators.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_validators[:max_validators]

def main():
    """Consensus demonstration"""
    print("=" * 60)
    print(" QENEX CONSENSUS - BFT + PoS")
    print("=" * 60)
    
    # Create mock components
    class MockBlockchain:
        def __init__(self):
            self.chain = []
            self.mempool = type('', (), {'get_transactions': lambda: []})()
    
    class MockNetwork:
        async def broadcast(self, msg):
            print(f"Broadcasting: {msg['type']}")
    
    blockchain = MockBlockchain()
    network = MockNetwork()
    
    # Create consensus engine
    consensus = BFTConsensus(
        node_id=hashlib.sha256(b"validator1").hexdigest(),
        blockchain=blockchain,
        network=network
    )
    
    # Generate keys
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    
    # Initialize as validator
    consensus.initialize_validator(private_key, Decimal('50000'))
    
    # Add more validators
    for i in range(2, 6):
        validator_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        consensus.add_validator(
            address=hashlib.sha256(f"validator{i}".encode()).hexdigest(),
            stake=Decimal(random.randint(10000, 100000)),
            public_key=validator_key.public_key()
        )
    
    consensus.update_active_validators()
    
    # Get consensus info
    info = consensus.get_consensus_info()
    
    print(f"\n[✓] Consensus initialized:")
    print(f"    Active validators: {info['validators']}")
    print(f"    Total stake: {info['total_stake']}")
    print(f"    Current phase: {info['phase']}")
    
    # Proof of Stake
    pos = ProofOfStake()
    
    # Add stakers
    pos.stake("validator1", Decimal('50000'))
    pos.stake("validator2", Decimal('30000'))
    pos.delegate("user1", "validator1", Decimal('10000'))
    
    # Calculate rewards
    rewards = pos.calculate_rewards(Decimal('100'))
    
    print(f"\n[💰] Staking rewards distribution:")
    for validator, reward in rewards.items():
        print(f"    {validator[:8]}...: {reward:.2f} tokens")
    
    print("\n" + "=" * 60)
    print(" CONSENSUS MECHANISM READY")
    print("=" * 60)

if __name__ == "__main__":
    main()