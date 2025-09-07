#!/usr/bin/env python3
"""
QENEX Payment Protocols - Multi-Network Payment Gateway
Support for SWIFT, SEPA, ACH, FedWire, and Card Networks
"""

import asyncio
import hashlib
import json
import logging
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import aiohttp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.serialization import load_pem_private_key

logger = logging.getLogger(__name__)


class PaymentNetwork(Enum):
    """Supported payment networks"""
    SWIFT = "SWIFT"
    SEPA = "SEPA"
    ACH = "ACH"
    FEDWIRE = "FEDWIRE"
    VISA = "VISA"
    MASTERCARD = "MASTERCARD"
    AMEX = "AMEX"
    INTERNAL = "INTERNAL"


class PaymentStatus(Enum):
    """Payment processing status"""
    INITIATED = "INITIATED"
    VALIDATED = "VALIDATED"
    PROCESSING = "PROCESSING"
    CLEARED = "CLEARED"
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    REVERSED = "REVERSED"


class ComplianceLevel(Enum):
    """Compliance check levels"""
    STANDARD = "STANDARD"
    ENHANCED = "ENHANCED"
    STRICT = "STRICT"


@dataclass
class PaymentInstruction:
    """Universal payment instruction"""
    id: UUID = field(default_factory=uuid4)
    network: PaymentNetwork = PaymentNetwork.INTERNAL
    source_account: str = ""
    source_routing: str = ""
    destination_account: str = ""
    destination_routing: str = ""
    amount: Decimal = Decimal("0.00")
    currency: str = "USD"
    reference: str = ""
    message: str = ""
    status: PaymentStatus = PaymentStatus.INITIATED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    compliance_data: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate payment instruction"""
        if self.amount <= 0:
            raise ValueError("Payment amount must be positive")
        
        if not self.source_account or not self.destination_account:
            raise ValueError("Source and destination accounts required")
            
        # Network-specific validation
        if self.network == PaymentNetwork.SWIFT:
            return self._validate_swift()
        elif self.network == PaymentNetwork.SEPA:
            return self._validate_sepa()
        elif self.network == PaymentNetwork.ACH:
            return self._validate_ach()
        elif self.network == PaymentNetwork.FEDWIRE:
            return self._validate_fedwire()
            
        return True
        
    def _validate_swift(self) -> bool:
        """Validate SWIFT payment"""
        # BIC validation (8 or 11 characters)
        bic_pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
        if not re.match(bic_pattern, self.source_routing):
            raise ValueError("Invalid source BIC")
        if not re.match(bic_pattern, self.destination_routing):
            raise ValueError("Invalid destination BIC")
        return True
        
    def _validate_sepa(self) -> bool:
        """Validate SEPA payment"""
        # IBAN validation
        iban_pattern = r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$'
        if not re.match(iban_pattern, self.destination_account):
            raise ValueError("Invalid IBAN")
        
        # Check SEPA currency
        if self.currency not in ['EUR']:
            raise ValueError("SEPA only supports EUR")
        
        # Amount limits
        if self.network == PaymentNetwork.SEPA and self.amount > Decimal("999999999.99"):
            raise ValueError("SEPA amount exceeds maximum")
            
        return True
        
    def _validate_ach(self) -> bool:
        """Validate ACH payment"""
        # Routing number validation (9 digits)
        if not re.match(r'^\d{9}$', self.source_routing):
            raise ValueError("Invalid source routing number")
        if not re.match(r'^\d{9}$', self.destination_routing):
            raise ValueError("Invalid destination routing number")
            
        # Check ACH currency
        if self.currency not in ['USD']:
            raise ValueError("ACH only supports USD")
            
        return True
        
    def _validate_fedwire(self) -> bool:
        """Validate FedWire payment"""
        # Similar to ACH but with different limits
        if not re.match(r'^\d{9}$', self.destination_routing):
            raise ValueError("Invalid FedWire routing number")
            
        if self.currency not in ['USD']:
            raise ValueError("FedWire only supports USD")
            
        return True


class NetworkAdapter:
    """Base class for network-specific adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize network adapter"""
        self.session = aiohttp.ClientSession()
        
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process payment through network"""
        raise NotImplementedError
        
    async def get_status(self, payment_id: str) -> PaymentStatus:
        """Get payment status from network"""
        raise NotImplementedError
        
    async def close(self):
        """Close network connections"""
        if self.session:
            await self.session.close()


class SWIFTAdapter(NetworkAdapter):
    """SWIFT network adapter"""
    
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process SWIFT payment"""
        # Format MT103 message
        mt103 = self._format_mt103(instruction)
        
        # In production, this would connect to SWIFT gateway
        # For now, simulate processing
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Log the message
        logger.info(f"SWIFT MT103 sent: {instruction.id}")
        
        return PaymentStatus.PROCESSING
        
    def _format_mt103(self, instruction: PaymentInstruction) -> str:
        """Format MT103 SWIFT message"""
        mt103 = f"""
{{1:F01{instruction.source_routing}0000000000}}
{{2:I103{instruction.destination_routing}N}}
{{3:{{108:MT103}}}}
{{4:
:20:{instruction.reference or str(instruction.id)[:16]}
:23B:CRED
:32A:{instruction.timestamp.strftime('%y%m%d')}{instruction.currency}{instruction.amount}
:50K:/{instruction.source_account}
:59:/{instruction.destination_account}
:70:{instruction.message[:140] if instruction.message else 'Payment'}
:71A:OUR
-}}
"""
        return mt103
        
    async def get_status(self, payment_id: str) -> PaymentStatus:
        """Get SWIFT payment status"""
        # In production, query SWIFT gpi tracker
        return PaymentStatus.PROCESSING


class SEPAAdapter(NetworkAdapter):
    """SEPA network adapter"""
    
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process SEPA payment"""
        # Create SEPA XML message (ISO 20022)
        sepa_xml = self._create_sepa_xml(instruction)
        
        # Simulate SEPA processing
        await asyncio.sleep(0.05)
        
        logger.info(f"SEPA payment initiated: {instruction.id}")
        
        return PaymentStatus.PROCESSING
        
    def _create_sepa_xml(self, instruction: PaymentInstruction) -> str:
        """Create SEPA XML message"""
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.03">
  <CstmrCdtTrfInitn>
    <GrpHdr>
      <MsgId>{instruction.id}</MsgId>
      <CreDtTm>{instruction.timestamp.isoformat()}</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <CtrlSum>{instruction.amount}</CtrlSum>
    </GrpHdr>
    <PmtInf>
      <PmtInfId>{instruction.id}</PmtInfId>
      <PmtMtd>TRF</PmtMtd>
      <CdtTrfTxInf>
        <PmtId>
          <EndToEndId>{instruction.reference or instruction.id}</EndToEndId>
        </PmtId>
        <Amt>
          <InstdAmt Ccy="{instruction.currency}">{instruction.amount}</InstdAmt>
        </Amt>
        <CdtrAcct>
          <Id>
            <IBAN>{instruction.destination_account}</IBAN>
          </Id>
        </CdtrAcct>
      </CdtTrfTxInf>
    </PmtInf>
  </CstmrCdtTrfInitn>
</Document>"""
        return xml


class ACHAdapter(NetworkAdapter):
    """ACH network adapter"""
    
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process ACH payment"""
        # Format NACHA file entry
        nacha_entry = self._format_nacha_entry(instruction)
        
        # Simulate ACH batch processing
        await asyncio.sleep(0.02)
        
        logger.info(f"ACH payment batched: {instruction.id}")
        
        return PaymentStatus.PROCESSING
        
    def _format_nacha_entry(self, instruction: PaymentInstruction) -> str:
        """Format NACHA file entry"""
        # Simplified NACHA format
        entry = f"""
6{instruction.metadata.get('transaction_code', '22')}
{instruction.destination_routing:>9}
{instruction.destination_account[:17]:>17}
{int(instruction.amount * 100):010d}
{instruction.reference[:15]:<15}
{instruction.metadata.get('receiver_name', 'RECEIVER')[:22]:<22}
{instruction.metadata.get('discretionary_data', '  '):<2}
0
{instruction.source_routing:>9}
"""
        return entry


class FedWireAdapter(NetworkAdapter):
    """FedWire network adapter"""
    
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process FedWire payment"""
        # Format FedWire message
        fedwire_msg = self._format_fedwire_message(instruction)
        
        # Simulate real-time gross settlement
        await asyncio.sleep(0.01)
        
        logger.info(f"FedWire payment sent: {instruction.id}")
        
        return PaymentStatus.CLEARED
        
    def _format_fedwire_message(self, instruction: PaymentInstruction) -> Dict[str, Any]:
        """Format FedWire message"""
        return {
            'message_type': '10',
            'sender_reference': str(instruction.id),
            'amount': str(instruction.amount),
            'sender_routing': instruction.source_routing,
            'receiver_routing': instruction.destination_routing,
            'receiver_account': instruction.destination_account,
            'business_function_code': '1030',
            'sender_to_receiver_info': instruction.message
        }


class CardNetworkAdapter(NetworkAdapter):
    """Card network adapter for Visa/Mastercard/Amex"""
    
    async def process_payment(self, instruction: PaymentInstruction) -> PaymentStatus:
        """Process card payment"""
        # Validate card details
        if not self._validate_card(instruction.source_account):
            return PaymentStatus.REJECTED
            
        # Simulate authorization
        auth_result = await self._authorize_payment(instruction)
        
        if auth_result:
            return PaymentStatus.CLEARED
        else:
            return PaymentStatus.REJECTED
            
    def _validate_card(self, card_number: str) -> bool:
        """Validate card number using Luhn algorithm"""
        if not card_number.isdigit():
            return False
            
        digits = [int(d) for d in card_number]
        checksum = 0
        
        for i in range(len(digits) - 2, -1, -2):
            doubled = digits[i] * 2
            if doubled > 9:
                doubled = doubled - 9
            digits[i] = doubled
            
        return sum(digits) % 10 == 0
        
    async def _authorize_payment(self, instruction: PaymentInstruction) -> bool:
        """Authorize card payment"""
        # Simulate authorization with issuer
        await asyncio.sleep(0.05)
        
        # Simple authorization logic
        if instruction.amount > Decimal("10000"):
            # Large transactions require additional checks
            return instruction.compliance_data.get('verified', False)
            
        return True


class ComplianceEngine:
    """Payment compliance and regulatory checks"""
    
    def __init__(self):
        self.aml_threshold = Decimal("10000")
        self.sanctions_list: Set[str] = set()
        self.high_risk_countries = {'IR', 'KP', 'SY'}
        
    async def check_compliance(
        self,
        instruction: PaymentInstruction,
        level: ComplianceLevel = ComplianceLevel.STANDARD
    ) -> Tuple[bool, Optional[str]]:
        """Run compliance checks on payment"""
        
        # Sanctions screening
        if await self._check_sanctions(instruction):
            return False, "Sanctions match found"
            
        # AML checks
        if instruction.amount >= self.aml_threshold:
            if not await self._perform_aml_checks(instruction):
                return False, "AML checks failed"
                
        # Country risk checks
        if level == ComplianceLevel.ENHANCED or level == ComplianceLevel.STRICT:
            country_code = instruction.metadata.get('destination_country')
            if country_code in self.high_risk_countries:
                if level == ComplianceLevel.STRICT:
                    return False, f"High-risk country: {country_code}"
                else:
                    # Enhanced due diligence required
                    instruction.compliance_data['enhanced_dd_required'] = True
                    
        # Transaction monitoring
        if not await self._monitor_transaction_patterns(instruction):
            return False, "Suspicious transaction pattern detected"
            
        return True, None
        
    async def _check_sanctions(self, instruction: PaymentInstruction) -> bool:
        """Check against sanctions lists"""
        # In production, check OFAC, EU, UN sanctions lists
        entities_to_check = [
            instruction.source_account,
            instruction.destination_account,
            instruction.metadata.get('sender_name', ''),
            instruction.metadata.get('receiver_name', '')
        ]
        
        for entity in entities_to_check:
            if entity in self.sanctions_list:
                return True
                
        return False
        
    async def _perform_aml_checks(self, instruction: PaymentInstruction) -> bool:
        """Perform AML checks"""
        # Check for structuring
        if await self._detect_structuring(instruction):
            return False
            
        # Check velocity
        if await self._check_velocity_limits(instruction):
            return False
            
        return True
        
    async def _detect_structuring(self, instruction: PaymentInstruction) -> bool:
        """Detect payment structuring attempts"""
        # Check for multiple payments just below threshold
        # In production, would query transaction history
        return False
        
    async def _check_velocity_limits(self, instruction: PaymentInstruction) -> bool:
        """Check transaction velocity limits"""
        # In production, check daily/weekly/monthly limits
        return False
        
    async def _monitor_transaction_patterns(self, instruction: PaymentInstruction) -> bool:
        """Monitor for suspicious transaction patterns"""
        # ML-based pattern detection would go here
        return True
        
    async def generate_reports(self, instruction: PaymentInstruction) -> Dict[str, Any]:
        """Generate regulatory reports"""
        reports = {}
        
        # CTR (Currency Transaction Report) for large cash transactions
        if instruction.amount >= Decimal("10000") and instruction.metadata.get('cash', False):
            reports['ctr'] = self._generate_ctr(instruction)
            
        # SAR (Suspicious Activity Report) if flagged
        if instruction.compliance_data.get('suspicious', False):
            reports['sar'] = self._generate_sar(instruction)
            
        return reports
        
    def _generate_ctr(self, instruction: PaymentInstruction) -> Dict[str, Any]:
        """Generate Currency Transaction Report"""
        return {
            'report_type': 'CTR',
            'transaction_id': str(instruction.id),
            'amount': str(instruction.amount),
            'currency': instruction.currency,
            'date': instruction.timestamp.isoformat(),
            'parties': {
                'sender': instruction.source_account,
                'receiver': instruction.destination_account
            }
        }
        
    def _generate_sar(self, instruction: PaymentInstruction) -> Dict[str, Any]:
        """Generate Suspicious Activity Report"""
        return {
            'report_type': 'SAR',
            'transaction_id': str(instruction.id),
            'suspicious_activity': instruction.compliance_data.get('suspicious_reason', 'Unknown'),
            'amount': str(instruction.amount),
            'date': instruction.timestamp.isoformat()
        }


class PaymentGateway:
    """Main payment gateway orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.adapters: Dict[PaymentNetwork, NetworkAdapter] = {}
        self.compliance = ComplianceEngine()
        self.payment_cache: Dict[UUID, PaymentInstruction] = {}
        
    async def initialize(self):
        """Initialize payment gateway"""
        # Initialize network adapters
        self.adapters[PaymentNetwork.SWIFT] = SWIFTAdapter(self.config)
        self.adapters[PaymentNetwork.SEPA] = SEPAAdapter(self.config)
        self.adapters[PaymentNetwork.ACH] = ACHAdapter(self.config)
        self.adapters[PaymentNetwork.FEDWIRE] = FedWireAdapter(self.config)
        
        # Initialize card networks
        card_adapter = CardNetworkAdapter(self.config)
        self.adapters[PaymentNetwork.VISA] = card_adapter
        self.adapters[PaymentNetwork.MASTERCARD] = card_adapter
        self.adapters[PaymentNetwork.AMEX] = card_adapter
        
        # Initialize all adapters
        for adapter in self.adapters.values():
            await adapter.initialize()
            
        logger.info("Payment gateway initialized")
        
    async def process_payment(
        self,
        instruction: PaymentInstruction,
        compliance_level: ComplianceLevel = ComplianceLevel.STANDARD
    ) -> Tuple[PaymentStatus, Optional[str]]:
        """Process payment through appropriate network"""
        
        try:
            # Validate instruction
            instruction.validate()
            
            # Run compliance checks
            compliant, reason = await self.compliance.check_compliance(instruction, compliance_level)
            if not compliant:
                instruction.status = PaymentStatus.REJECTED
                return PaymentStatus.REJECTED, reason
                
            # Store in cache
            self.payment_cache[instruction.id] = instruction
            
            # Route to appropriate network
            if instruction.network in self.adapters:
                adapter = self.adapters[instruction.network]
                status = await adapter.process_payment(instruction)
                instruction.status = status
                
                # Generate reports if needed
                if instruction.amount >= self.compliance.aml_threshold:
                    reports = await self.compliance.generate_reports(instruction)
                    if reports:
                        logger.info(f"Regulatory reports generated: {list(reports.keys())}")
                        
                return status, None
            else:
                return PaymentStatus.FAILED, f"Unsupported network: {instruction.network}"
                
        except Exception as e:
            logger.error(f"Payment processing error: {e}")
            return PaymentStatus.FAILED, str(e)
            
    async def get_payment_status(self, payment_id: UUID) -> Optional[PaymentStatus]:
        """Get payment status"""
        if payment_id in self.payment_cache:
            instruction = self.payment_cache[payment_id]
            
            # Check with network for updated status
            if instruction.network in self.adapters:
                adapter = self.adapters[instruction.network]
                status = await adapter.get_status(str(payment_id))
                instruction.status = status
                return status
                
            return instruction.status
            
        return None
        
    async def reverse_payment(self, payment_id: UUID, reason: str) -> bool:
        """Reverse a payment"""
        if payment_id not in self.payment_cache:
            return False
            
        instruction = self.payment_cache[payment_id]
        
        if instruction.status not in [PaymentStatus.SETTLED, PaymentStatus.CLEARED]:
            return False
            
        # Create reversal instruction
        reversal = PaymentInstruction(
            network=instruction.network,
            source_account=instruction.destination_account,
            source_routing=instruction.destination_routing,
            destination_account=instruction.source_account,
            destination_routing=instruction.source_routing,
            amount=instruction.amount,
            currency=instruction.currency,
            reference=f"REVERSAL-{instruction.id}",
            message=f"Reversal: {reason}",
            metadata={'original_payment': str(instruction.id), 'reversal_reason': reason}
        )
        
        status, _ = await self.process_payment(reversal)
        
        if status in [PaymentStatus.PROCESSING, PaymentStatus.CLEARED]:
            instruction.status = PaymentStatus.REVERSED
            return True
            
        return False
        
    async def close(self):
        """Close payment gateway"""
        for adapter in self.adapters.values():
            await adapter.close()
            
        logger.info("Payment gateway closed")
        
    def get_supported_networks(self) -> List[str]:
        """Get list of supported payment networks"""
        return [network.value for network in PaymentNetwork]
        
    def get_network_capabilities(self, network: PaymentNetwork) -> Dict[str, Any]:
        """Get capabilities of a payment network"""
        capabilities = {
            PaymentNetwork.SWIFT: {
                'currencies': 'All',
                'settlement_time': '1-5 days',
                'max_amount': None,
                'message_types': ['MT103', 'MT202', 'MT900']
            },
            PaymentNetwork.SEPA: {
                'currencies': ['EUR'],
                'settlement_time': '1 day (SCT Inst: instant)',
                'max_amount': 999999999.99,
                'schemes': ['SCT', 'SDD', 'SCT Inst']
            },
            PaymentNetwork.ACH: {
                'currencies': ['USD'],
                'settlement_time': '1-3 days (Same-day available)',
                'max_amount': None,
                'sec_codes': ['PPD', 'CCD', 'WEB', 'TEL']
            },
            PaymentNetwork.FEDWIRE: {
                'currencies': ['USD'],
                'settlement_time': 'Real-time',
                'max_amount': None,
                'operating_hours': 'Business days 9PM-7PM ET'
            }
        }
        
        return capabilities.get(network, {})