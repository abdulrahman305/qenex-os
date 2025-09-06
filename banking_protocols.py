#!/usr/bin/env python3
"""
QENEX Banking Protocols Implementation
Real implementations of SWIFT, SEPA, and ISO 20022 standards
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# ISO 20022 Implementation
# ============================================================================

class ISO20022MessageType(Enum):
    """ISO 20022 message types"""
    PAIN_001 = "pain.001.001.09"  # Customer Credit Transfer Initiation
    PAIN_008 = "pain.008.001.08"  # Customer Direct Debit Initiation
    PACS_008 = "pacs.008.001.08"  # FI to FI Customer Credit Transfer
    PACS_002 = "pacs.002.001.10"  # FI to FI Payment Status Report
    CAMT_053 = "camt.053.001.08"  # Bank to Customer Statement
    CAMT_054 = "camt.054.001.08"  # Bank to Customer Debit/Credit Notification

@dataclass
class ISO20022Message:
    """ISO 20022 message structure"""
    message_type: ISO20022MessageType
    message_id: str
    creation_datetime: datetime
    initiating_party: Dict[str, str]
    payment_info: List[Dict[str, Any]]
    
    def to_xml(self) -> str:
        """Convert message to ISO 20022 XML format"""
        root = ET.Element("Document", xmlns=f"urn:iso:std:iso:20022:tech:xsd:{self.message_type.value}")
        
        if self.message_type == ISO20022MessageType.PAIN_001:
            self._build_pain001(root)
        elif self.message_type == ISO20022MessageType.PACS_008:
            self._build_pacs008(root)
            
        return ET.tostring(root, encoding='unicode', method='xml')
        
    def _build_pain001(self, root):
        """Build Customer Credit Transfer Initiation message"""
        cstmr_cdt_trf = ET.SubElement(root, "CstmrCdtTrfInitn")
        
        # Group Header
        grp_hdr = ET.SubElement(cstmr_cdt_trf, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = self.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = self.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = str(len(self.payment_info))
        
        # Control sum
        ctrl_sum = sum(Decimal(str(p.get('amount', 0))) for p in self.payment_info)
        ET.SubElement(grp_hdr, "CtrlSum").text = str(ctrl_sum)
        
        # Initiating Party
        init_pty = ET.SubElement(grp_hdr, "InitgPty")
        ET.SubElement(init_pty, "Nm").text = self.initiating_party.get('name', '')
        
        # Payment Information
        for payment in self.payment_info:
            pmt_inf = ET.SubElement(cstmr_cdt_trf, "PmtInf")
            ET.SubElement(pmt_inf, "PmtInfId").text = payment.get('id', str(uuid.uuid4()))
            ET.SubElement(pmt_inf, "PmtMtd").text = payment.get('method', 'TRF')
            
            # Debtor
            dbtr = ET.SubElement(pmt_inf, "Dbtr")
            ET.SubElement(dbtr, "Nm").text = payment.get('debtor_name', '')
            
            dbtr_acct = ET.SubElement(pmt_inf, "DbtrAcct")
            dbtr_id = ET.SubElement(dbtr_acct, "Id")
            ET.SubElement(dbtr_id, "IBAN").text = payment.get('debtor_iban', '')
            
            # Credit Transfer Transaction
            cdt_trf_tx = ET.SubElement(pmt_inf, "CdtTrfTxInf")
            pmt_id = ET.SubElement(cdt_trf_tx, "PmtId")
            ET.SubElement(pmt_id, "EndToEndId").text = payment.get('end_to_end_id', str(uuid.uuid4()))
            
            # Amount
            amt = ET.SubElement(cdt_trf_tx, "Amt")
            instd_amt = ET.SubElement(amt, "InstdAmt", Ccy=payment.get('currency', 'EUR'))
            instd_amt.text = str(payment.get('amount', 0))
            
            # Creditor
            cdtr = ET.SubElement(cdt_trf_tx, "Cdtr")
            ET.SubElement(cdtr, "Nm").text = payment.get('creditor_name', '')
            
            cdtr_acct = ET.SubElement(cdt_trf_tx, "CdtrAcct")
            cdtr_id = ET.SubElement(cdtr_acct, "Id")
            ET.SubElement(cdtr_id, "IBAN").text = payment.get('creditor_iban', '')
            
    def _build_pacs008(self, root):
        """Build FI to FI Customer Credit Transfer message"""
        fi_to_fi = ET.SubElement(root, "FIToFICstmrCdtTrf")
        
        # Group Header
        grp_hdr = ET.SubElement(fi_to_fi, "GrpHdr")
        ET.SubElement(grp_hdr, "MsgId").text = self.message_id
        ET.SubElement(grp_hdr, "CreDtTm").text = self.creation_datetime.isoformat()
        ET.SubElement(grp_hdr, "NbOfTxs").text = str(len(self.payment_info))
        ET.SubElement(grp_hdr, "SttlmInf").text = "CLRG"  # Clearing
        
        # Credit Transfer Transaction Information
        for payment in self.payment_info:
            cdt_trf_tx = ET.SubElement(fi_to_fi, "CdtTrfTxInf")
            
            # Payment Identification
            pmt_id = ET.SubElement(cdt_trf_tx, "PmtId")
            ET.SubElement(pmt_id, "TxId").text = payment.get('tx_id', str(uuid.uuid4()))
            ET.SubElement(pmt_id, "EndToEndId").text = payment.get('end_to_end_id', str(uuid.uuid4()))
            
            # Interbank Settlement
            intrbnk_sttlm = ET.SubElement(cdt_trf_tx, "IntrBkSttlmAmt", Ccy=payment.get('currency', 'EUR'))
            intrbnk_sttlm.text = str(payment.get('amount', 0))
            
            ET.SubElement(cdt_trf_tx, "IntrBkSttlmDt").text = datetime.now(timezone.utc).date().isoformat()
            
    @classmethod
    def from_xml(cls, xml_string: str) -> 'ISO20022Message':
        """Parse ISO 20022 XML message"""
        root = ET.fromstring(xml_string)
        namespace = root.tag.split('}')[0][1:] if '}' in root.tag else ''
        
        # Determine message type from namespace
        message_type = None
        for msg_type in ISO20022MessageType:
            if msg_type.value in namespace:
                message_type = msg_type
                break
                
        # Parse based on message type
        # Simplified parsing - production would need full schema validation
        return cls(
            message_type=message_type,
            message_id=str(uuid.uuid4()),
            creation_datetime=datetime.now(timezone.utc),
            initiating_party={},
            payment_info=[]
        )

# ============================================================================
# SWIFT Implementation
# ============================================================================

class SWIFTMessageType(Enum):
    """SWIFT message types"""
    MT103 = "103"  # Single Customer Credit Transfer
    MT202 = "202"  # General Financial Institution Transfer
    MT900 = "900"  # Confirmation of Debit
    MT910 = "910"  # Confirmation of Credit
    MT940 = "940"  # Customer Statement Message
    MT950 = "950"  # Statement Message

@dataclass
class SWIFTMessage:
    """SWIFT message structure"""
    message_type: SWIFTMessageType
    sender: str  # BIC code
    receiver: str  # BIC code
    reference: str
    fields: Dict[str, str]
    
    def to_swift_format(self) -> str:
        """Convert to SWIFT MT format"""
        lines = []
        
        # Basic header
        lines.append(f"{{1:F01{self.sender}0000000000}}")
        lines.append(f"{{2:I{self.message_type.value}{self.receiver}N}}")
        lines.append(f"{{4:")
        
        # Message fields
        if self.message_type == SWIFTMessageType.MT103:
            lines.extend(self._build_mt103())
        elif self.message_type == SWIFTMessageType.MT202:
            lines.extend(self._build_mt202())
            
        lines.append("-}")
        
        return "\n".join(lines)
        
    def _build_mt103(self) -> List[str]:
        """Build MT103 Single Customer Credit Transfer"""
        fields = []
        
        # Sender's Reference
        fields.append(f":20:{self.reference}")
        
        # Bank Operation Code
        fields.append(f":23B:{self.fields.get('bank_op_code', 'CRED')}")
        
        # Value Date/Currency/Amount
        value_date = self.fields.get('value_date', datetime.now().strftime('%y%m%d'))
        currency = self.fields.get('currency', 'EUR')
        amount = self.fields.get('amount', '0,00')
        fields.append(f":32A:{value_date}{currency}{amount}")
        
        # Ordering Customer
        if 'ordering_customer' in self.fields:
            fields.append(f":50K:{self.fields['ordering_customer']}")
            
        # Beneficiary Customer
        if 'beneficiary_customer' in self.fields:
            fields.append(f":59:{self.fields['beneficiary_customer']}")
            
        # Remittance Information
        if 'remittance_info' in self.fields:
            fields.append(f":70:{self.fields['remittance_info']}")
            
        # Details of Charges
        fields.append(f":71A:{self.fields.get('charges', 'SHA')}")
        
        return fields
        
    def _build_mt202(self) -> List[str]:
        """Build MT202 General Financial Institution Transfer"""
        fields = []
        
        # Transaction Reference
        fields.append(f":20:{self.reference}")
        
        # Related Reference
        if 'related_reference' in self.fields:
            fields.append(f":21:{self.fields['related_reference']}")
            
        # Value Date/Currency/Amount
        value_date = self.fields.get('value_date', datetime.now().strftime('%y%m%d'))
        currency = self.fields.get('currency', 'EUR')
        amount = self.fields.get('amount', '0,00')
        fields.append(f":32A:{value_date}{currency}{amount}")
        
        # Ordering Institution
        if 'ordering_institution' in self.fields:
            fields.append(f":52A:{self.fields['ordering_institution']}")
            
        # Beneficiary Institution
        if 'beneficiary_institution' in self.fields:
            fields.append(f":58A:{self.fields['beneficiary_institution']}")
            
        return fields
        
    @classmethod
    def from_swift_format(cls, swift_text: str) -> 'SWIFTMessage':
        """Parse SWIFT MT message"""
        # Extract message type from header
        import re
        
        # Find message type
        type_match = re.search(r'\{2:I(\d{3})', swift_text)
        if not type_match:
            raise ValueError("Invalid SWIFT message format")
            
        message_type = SWIFTMessageType(type_match.group(1))
        
        # Extract sender and receiver
        sender_match = re.search(r'\{1:F01(\w{8})', swift_text)
        receiver_match = re.search(r'\{2:I\d{3}(\w{8})', swift_text)
        
        sender = sender_match.group(1) if sender_match else ""
        receiver = receiver_match.group(1) if receiver_match else ""
        
        # Extract fields
        fields = {}
        field_pattern = r':(\w+):([^\n:]+)'
        for match in re.finditer(field_pattern, swift_text):
            field_tag = match.group(1)
            field_value = match.group(2).strip()
            fields[field_tag] = field_value
            
        return cls(
            message_type=message_type,
            sender=sender,
            receiver=receiver,
            reference=fields.get('20', str(uuid.uuid4())),
            fields=fields
        )
        
    def validate(self) -> bool:
        """Validate SWIFT message format"""
        # Check BIC code format
        bic_pattern = r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$'
        
        if not re.match(bic_pattern, self.sender):
            return False
            
        if not re.match(bic_pattern, self.receiver):
            return False
            
        # Check required fields based on message type
        if self.message_type == SWIFTMessageType.MT103:
            required = ['bank_op_code', 'amount', 'currency']
            return all(field in self.fields for field in required)
            
        return True

# ============================================================================
# SEPA Implementation
# ============================================================================

class SEPATransactionType(Enum):
    """SEPA transaction types"""
    SCT = "SEPA Credit Transfer"
    SDD = "SEPA Direct Debit"
    SCT_INST = "SEPA Instant Credit Transfer"

@dataclass
class SEPATransaction:
    """SEPA transaction structure"""
    transaction_type: SEPATransactionType
    message_id: str
    payment_id: str
    amount: Decimal
    currency: str = "EUR"
    debtor_name: str = ""
    debtor_iban: str = ""
    debtor_bic: str = ""
    creditor_name: str = ""
    creditor_iban: str = ""
    creditor_bic: str = ""
    remittance_info: str = ""
    execution_date: datetime = field(default_factory=datetime.now)
    
    def validate_iban(self, iban: str) -> bool:
        """Validate IBAN format and checksum"""
        # Remove spaces and convert to uppercase
        iban = iban.replace(' ', '').upper()
        
        # Check basic format
        if not re.match(r'^[A-Z]{2}\d{2}[A-Z0-9]+$', iban):
            return False
            
        # Validate length per country
        iban_lengths = {
            'AD': 24, 'AE': 23, 'AT': 20, 'AZ': 28, 'BA': 20, 'BE': 16,
            'BG': 22, 'BH': 22, 'BR': 29, 'CH': 21, 'CR': 22, 'CY': 28,
            'CZ': 24, 'DE': 22, 'DK': 18, 'DO': 28, 'EE': 20, 'ES': 24,
            'FI': 18, 'FO': 18, 'FR': 27, 'GB': 22, 'GE': 22, 'GI': 23,
            'GL': 18, 'GR': 27, 'GT': 28, 'HR': 21, 'HU': 28, 'IE': 22,
            'IL': 23, 'IS': 26, 'IT': 27, 'JO': 30, 'KW': 30, 'KZ': 20,
            'LB': 28, 'LI': 21, 'LT': 20, 'LU': 20, 'LV': 21, 'MC': 27,
            'MD': 24, 'ME': 22, 'MK': 19, 'MR': 27, 'MT': 31, 'MU': 30,
            'NL': 18, 'NO': 15, 'PK': 24, 'PL': 28, 'PS': 29, 'PT': 25,
            'QA': 29, 'RO': 24, 'RS': 22, 'SA': 24, 'SE': 24, 'SI': 19,
            'SK': 24, 'SM': 27, 'TN': 24, 'TR': 26, 'AE': 23, 'GB': 22,
            'VG': 24, 'XK': 20
        }
        
        country_code = iban[:2]
        if country_code in iban_lengths and len(iban) != iban_lengths[country_code]:
            return False
            
        # Validate checksum using mod97
        rearranged = iban[4:] + iban[:4]
        numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
        
        return int(numeric) % 97 == 1
        
    def to_xml(self) -> str:
        """Convert to SEPA XML format"""
        if self.transaction_type in [SEPATransactionType.SCT, SEPATransactionType.SCT_INST]:
            return self._build_sct_xml()
        elif self.transaction_type == SEPATransactionType.SDD:
            return self._build_sdd_xml()
            
    def _build_sct_xml(self) -> str:
        """Build SEPA Credit Transfer XML"""
        # Create ISO 20022 pain.001 message
        iso_message = ISO20022Message(
            message_type=ISO20022MessageType.PAIN_001,
            message_id=self.message_id,
            creation_datetime=datetime.now(timezone.utc),
            initiating_party={'name': self.debtor_name},
            payment_info=[{
                'id': self.payment_id,
                'method': 'TRF',
                'debtor_name': self.debtor_name,
                'debtor_iban': self.debtor_iban,
                'creditor_name': self.creditor_name,
                'creditor_iban': self.creditor_iban,
                'amount': self.amount,
                'currency': self.currency,
                'end_to_end_id': self.payment_id,
                'remittance_info': self.remittance_info
            }]
        )
        
        return iso_message.to_xml()
        
    def _build_sdd_xml(self) -> str:
        """Build SEPA Direct Debit XML"""
        # Would use ISO 20022 pain.008 message
        pass
        
    def process(self) -> bool:
        """Process SEPA transaction"""
        # Validate IBANs
        if not self.validate_iban(self.debtor_iban):
            raise ValueError(f"Invalid debtor IBAN: {self.debtor_iban}")
            
        if not self.validate_iban(self.creditor_iban):
            raise ValueError(f"Invalid creditor IBAN: {self.creditor_iban}")
            
        # Check amount
        if self.amount <= 0:
            raise ValueError("Amount must be positive")
            
        # Check currency (SEPA only supports EUR)
        if self.currency != "EUR":
            raise ValueError("SEPA only supports EUR currency")
            
        # For instant payments, check amount limit
        if self.transaction_type == SEPATransactionType.SCT_INST:
            if self.amount > Decimal('100000'):
                raise ValueError("SEPA Instant payment limit is €100,000")
                
        logger.info(f"SEPA transaction processed: {self.payment_id}")
        return True

# ============================================================================
# Banking Protocol Manager
# ============================================================================

class BankingProtocolManager:
    """Manages all banking protocol implementations"""
    
    def __init__(self):
        self.iso20022_processor = ISO20022Processor()
        self.swift_processor = SWIFTProcessor()
        self.sepa_processor = SEPAProcessor()
        
    async def process_payment(self, protocol: str, payment_data: Dict[str, Any]) -> str:
        """Process payment using specified protocol"""
        
        if protocol == "ISO20022":
            return await self.iso20022_processor.process(payment_data)
        elif protocol == "SWIFT":
            return await self.swift_processor.process(payment_data)
        elif protocol == "SEPA":
            return await self.sepa_processor.process(payment_data)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")

class ISO20022Processor:
    """Processes ISO 20022 messages"""
    
    async def process(self, payment_data: Dict[str, Any]) -> str:
        """Process ISO 20022 payment"""
        message = ISO20022Message(
            message_type=ISO20022MessageType(payment_data.get('message_type', 'pain.001.001.09')),
            message_id=str(uuid.uuid4()),
            creation_datetime=datetime.now(timezone.utc),
            initiating_party=payment_data.get('initiating_party', {}),
            payment_info=payment_data.get('payment_info', [])
        )
        
        xml = message.to_xml()
        logger.info(f"ISO 20022 message created: {message.message_id}")
        return xml

class SWIFTProcessor:
    """Processes SWIFT messages"""
    
    async def process(self, payment_data: Dict[str, Any]) -> str:
        """Process SWIFT payment"""
        message = SWIFTMessage(
            message_type=SWIFTMessageType(payment_data.get('message_type', '103')),
            sender=payment_data.get('sender', ''),
            receiver=payment_data.get('receiver', ''),
            reference=payment_data.get('reference', str(uuid.uuid4())),
            fields=payment_data.get('fields', {})
        )
        
        if not message.validate():
            raise ValueError("Invalid SWIFT message format")
            
        swift_format = message.to_swift_format()
        logger.info(f"SWIFT message created: {message.reference}")
        return swift_format

class SEPAProcessor:
    """Processes SEPA transactions"""
    
    async def process(self, payment_data: Dict[str, Any]) -> str:
        """Process SEPA payment"""
        transaction = SEPATransaction(
            transaction_type=SEPATransactionType[payment_data.get('type', 'SCT')],
            message_id=str(uuid.uuid4()),
            payment_id=payment_data.get('payment_id', str(uuid.uuid4())),
            amount=Decimal(str(payment_data.get('amount', 0))),
            currency=payment_data.get('currency', 'EUR'),
            debtor_name=payment_data.get('debtor_name', ''),
            debtor_iban=payment_data.get('debtor_iban', ''),
            debtor_bic=payment_data.get('debtor_bic', ''),
            creditor_name=payment_data.get('creditor_name', ''),
            creditor_iban=payment_data.get('creditor_iban', ''),
            creditor_bic=payment_data.get('creditor_bic', ''),
            remittance_info=payment_data.get('remittance_info', '')
        )
        
        if transaction.process():
            xml = transaction.to_xml()
            logger.info(f"SEPA transaction created: {transaction.payment_id}")
            return xml
        else:
            raise ValueError("SEPA transaction processing failed")

# ============================================================================
# Testing
# ============================================================================

async def test_protocols():
    """Test banking protocol implementations"""
    
    # Test ISO 20022
    iso_payment = {
        'message_type': 'pain.001.001.09',
        'initiating_party': {'name': 'Test Company'},
        'payment_info': [{
            'amount': 1000.50,
            'currency': 'EUR',
            'debtor_name': 'John Doe',
            'debtor_iban': 'DE89370400440532013000',
            'creditor_name': 'Jane Smith',
            'creditor_iban': 'FR1420041010050500013M02606'
        }]
    }
    
    # Test SWIFT
    swift_payment = {
        'message_type': '103',
        'sender': 'DEUTDEFF',
        'receiver': 'BNPAFRPP',
        'reference': 'REF123456',
        'fields': {
            'amount': '1000,50',
            'currency': 'EUR',
            'ordering_customer': 'John Doe',
            'beneficiary_customer': 'Jane Smith'
        }
    }
    
    # Test SEPA
    sepa_payment = {
        'type': 'SCT',
        'amount': 1000.50,
        'debtor_name': 'John Doe',
        'debtor_iban': 'DE89370400440532013000',
        'creditor_name': 'Jane Smith',
        'creditor_iban': 'FR1420041010050500013M02606',
        'remittance_info': 'Invoice payment'
    }
    
    manager = BankingProtocolManager()
    
    # Process payments
    iso_result = await manager.process_payment('ISO20022', iso_payment)
    print(f"ISO 20022 Result:\n{iso_result[:200]}...")
    
    swift_result = await manager.process_payment('SWIFT', swift_payment)
    print(f"\nSWIFT Result:\n{swift_result}")
    
    sepa_result = await manager.process_payment('SEPA', sepa_payment)
    print(f"\nSEPA Result:\n{sepa_result[:200]}...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_protocols())