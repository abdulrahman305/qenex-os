#!/usr/bin/env python3
"""
QENEX Payment Protocols Implementation
Support for SWIFT, SEPA, ACH, FedWire, and other payment networks
"""

import asyncio
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import aiohttp
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa


class PaymentNetwork(Enum):
    """Supported payment networks"""
    SWIFT = auto()
    SEPA = auto()
    ACH = auto()
    FEDWIRE = auto()
    FASTER_PAYMENTS = auto()
    RTGS = auto()
    TARGET2 = auto()
    CHIPS = auto()


class MessageType(Enum):
    """Payment message types"""
    # SWIFT Messages
    MT103 = "Customer Transfer"
    MT202 = "Bank Transfer"
    MT202COV = "Cover Payment"
    MT900 = "Debit Confirmation"
    MT910 = "Credit Confirmation"
    MT940 = "Account Statement"
    MT950 = "Statement Message"
    
    # ISO 20022 Messages
    PAIN001 = "Customer Credit Transfer Initiation"
    PAIN008 = "Customer Direct Debit Initiation"
    PACS008 = "Financial Institution Credit Transfer"
    PACS009 = "Financial Institution Debit Transfer"
    CAMT053 = "Bank to Customer Statement"
    CAMT054 = "Bank to Customer Debit/Credit Notification"


@dataclass
class PaymentMessage:
    """Universal payment message structure"""
    id: str
    network: PaymentNetwork
    message_type: MessageType
    sender: str
    receiver: str
    amount: Decimal
    currency: str
    value_date: datetime
    reference: str
    metadata: Dict[str, Any]
    raw_message: Optional[str] = None
    signature: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'network': self.network.name,
            'message_type': self.message_type.name,
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': str(self.amount),
            'currency': self.currency,
            'value_date': self.value_date.isoformat(),
            'reference': self.reference,
            'metadata': self.metadata,
            'signature': self.signature
        }


class SWIFTProtocol:
    """SWIFT payment protocol implementation"""
    
    def __init__(self):
        self.bic_pattern = re.compile(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$')
        self.iban_pattern = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$')
        
    def validate_bic(self, bic: str) -> bool:
        """Validate SWIFT BIC code"""
        return bool(self.bic_pattern.match(bic))
    
    def validate_iban(self, iban: str) -> bool:
        """Validate IBAN"""
        if not self.iban_pattern.match(iban):
            return False
        
        # Validate IBAN checksum
        iban_digits = (iban[4:] + iban[:4]).upper()
        iban_number = ''.join(str(ord(c) - 55) if c.isalpha() else c for c in iban_digits)
        return int(iban_number) % 97 == 1
    
    def create_mt103(
        self,
        sender_bic: str,
        receiver_bic: str,
        sender_account: str,
        receiver_account: str,
        amount: Decimal,
        currency: str,
        reference: str,
        remittance_info: str = ""
    ) -> PaymentMessage:
        """Create MT103 customer transfer message"""
        
        if not self.validate_bic(sender_bic):
            raise ValueError(f"Invalid sender BIC: {sender_bic}")
        if not self.validate_bic(receiver_bic):
            raise ValueError(f"Invalid receiver BIC: {receiver_bic}")
        
        message_id = f"MT103-{uuid4().hex[:16].upper()}"
        
        # Build MT103 message format
        mt103_fields = {
            ':20:': reference,  # Transaction Reference
            ':23B:': 'CRED',  # Bank Operation Code
            ':32A:': f"{datetime.now().strftime('%y%m%d')}{currency}{amount}",  # Value Date/Currency/Amount
            ':50K:': f"/{sender_account}\n{sender_bic}",  # Ordering Customer
            ':59:': f"/{receiver_account}\n{receiver_bic}",  # Beneficiary Customer
            ':70:': remittance_info[:140],  # Remittance Information
            ':71A:': 'OUR',  # Details of Charges
        }
        
        raw_message = '\n'.join(f"{k}{v}" for k, v in mt103_fields.items())
        
        return PaymentMessage(
            id=message_id,
            network=PaymentNetwork.SWIFT,
            message_type=MessageType.MT103,
            sender=sender_bic,
            receiver=receiver_bic,
            amount=amount,
            currency=currency,
            value_date=datetime.now(timezone.utc),
            reference=reference,
            metadata={
                'sender_account': sender_account,
                'receiver_account': receiver_account,
                'remittance_info': remittance_info,
                'charges': 'OUR'
            },
            raw_message=raw_message
        )
    
    def create_mt202(
        self,
        sender_bic: str,
        receiver_bic: str,
        amount: Decimal,
        currency: str,
        reference: str
    ) -> PaymentMessage:
        """Create MT202 bank-to-bank transfer message"""
        
        if not self.validate_bic(sender_bic):
            raise ValueError(f"Invalid sender BIC: {sender_bic}")
        if not self.validate_bic(receiver_bic):
            raise ValueError(f"Invalid receiver BIC: {receiver_bic}")
        
        message_id = f"MT202-{uuid4().hex[:16].upper()}"
        
        # Build MT202 message format
        mt202_fields = {
            ':20:': reference,
            ':21:': reference,  # Related Reference
            ':32A:': f"{datetime.now().strftime('%y%m%d')}{currency}{amount}",
            ':52A:': sender_bic,  # Ordering Institution
            ':58A:': receiver_bic,  # Beneficiary Institution
        }
        
        raw_message = '\n'.join(f"{k}{v}" for k, v in mt202_fields.items())
        
        return PaymentMessage(
            id=message_id,
            network=PaymentNetwork.SWIFT,
            message_type=MessageType.MT202,
            sender=sender_bic,
            receiver=receiver_bic,
            amount=amount,
            currency=currency,
            value_date=datetime.now(timezone.utc),
            reference=reference,
            metadata={'type': 'bank_transfer'},
            raw_message=raw_message
        )
    
    def parse_mt103(self, raw_message: str) -> Dict[str, Any]:
        """Parse MT103 message"""
        parsed = {}
        lines = raw_message.split('\n')
        
        for line in lines:
            if line.startswith(':20:'):
                parsed['reference'] = line[4:]
            elif line.startswith(':32A:'):
                value = line[5:]
                parsed['value_date'] = value[:6]
                parsed['currency'] = value[6:9]
                parsed['amount'] = value[9:].replace(',', '.')
            elif line.startswith(':50K:'):
                parsed['ordering_customer'] = line[5:]
            elif line.startswith(':59:'):
                parsed['beneficiary'] = line[4:]
            elif line.startswith(':70:'):
                parsed['remittance_info'] = line[4:]
                
        return parsed


class SEPAProtocol:
    """SEPA (Single Euro Payments Area) protocol implementation"""
    
    def __init__(self):
        self.supported_currencies = {'EUR'}
        self.sepa_countries = {
            'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
            'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU',
            'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE',
            'CH', 'GB', 'SM', 'VA', 'MC', 'AD'
        }
    
    def validate_sepa_account(self, iban: str) -> bool:
        """Validate SEPA account (IBAN)"""
        if len(iban) < 15 or len(iban) > 34:
            return False
        
        country_code = iban[:2]
        return country_code in self.sepa_countries
    
    def create_sct(
        self,
        debtor_iban: str,
        debtor_name: str,
        creditor_iban: str,
        creditor_name: str,
        amount: Decimal,
        reference: str,
        remittance_info: str = ""
    ) -> PaymentMessage:
        """Create SEPA Credit Transfer (SCT)"""
        
        if not self.validate_sepa_account(debtor_iban):
            raise ValueError(f"Invalid debtor IBAN: {debtor_iban}")
        if not self.validate_sepa_account(creditor_iban):
            raise ValueError(f"Invalid creditor IBAN: {creditor_iban}")
        
        message_id = f"SCT-{uuid4().hex[:16].upper()}"
        
        # Create ISO 20022 pain.001 XML message
        root = ET.Element('Document', xmlns='urn:iso:std:iso:20022:tech:xsd:pain.001.001.03')
        cstmr_cdt_trf = ET.SubElement(root, 'CstmrCdtTrfInitn')
        
        # Group Header
        grp_hdr = ET.SubElement(cstmr_cdt_trf, 'GrpHdr')
        ET.SubElement(grp_hdr, 'MsgId').text = message_id
        ET.SubElement(grp_hdr, 'CreDtTm').text = datetime.now(timezone.utc).isoformat()
        ET.SubElement(grp_hdr, 'NbOfTxs').text = '1'
        ET.SubElement(grp_hdr, 'CtrlSum').text = str(amount)
        
        # Payment Information
        pmt_inf = ET.SubElement(cstmr_cdt_trf, 'PmtInf')
        ET.SubElement(pmt_inf, 'PmtInfId').text = message_id
        ET.SubElement(pmt_inf, 'PmtMtd').text = 'TRF'
        
        # Debtor
        dbtr = ET.SubElement(pmt_inf, 'Dbtr')
        ET.SubElement(dbtr, 'Nm').text = debtor_name
        dbtr_acct = ET.SubElement(pmt_inf, 'DbtrAcct')
        dbtr_id = ET.SubElement(dbtr_acct, 'Id')
        ET.SubElement(dbtr_id, 'IBAN').text = debtor_iban
        
        # Credit Transfer Transaction
        cdt_trf_tx = ET.SubElement(pmt_inf, 'CdtTrfTxInf')
        pmt_id = ET.SubElement(cdt_trf_tx, 'PmtId')
        ET.SubElement(pmt_id, 'EndToEndId').text = reference
        
        amt = ET.SubElement(cdt_trf_tx, 'Amt')
        instd_amt = ET.SubElement(amt, 'InstdAmt', Ccy='EUR')
        instd_amt.text = str(amount)
        
        # Creditor
        cdtr = ET.SubElement(cdt_trf_tx, 'Cdtr')
        ET.SubElement(cdtr, 'Nm').text = creditor_name
        cdtr_acct = ET.SubElement(cdt_trf_tx, 'CdtrAcct')
        cdtr_id = ET.SubElement(cdtr_acct, 'Id')
        ET.SubElement(cdtr_id, 'IBAN').text = creditor_iban
        
        # Remittance Information
        if remittance_info:
            rmt_inf = ET.SubElement(cdt_trf_tx, 'RmtInf')
            ET.SubElement(rmt_inf, 'Ustrd').text = remittance_info[:140]
        
        raw_message = ET.tostring(root, encoding='unicode')
        
        return PaymentMessage(
            id=message_id,
            network=PaymentNetwork.SEPA,
            message_type=MessageType.PAIN001,
            sender=debtor_iban,
            receiver=creditor_iban,
            amount=amount,
            currency='EUR',
            value_date=datetime.now(timezone.utc),
            reference=reference,
            metadata={
                'debtor_name': debtor_name,
                'creditor_name': creditor_name,
                'remittance_info': remittance_info,
                'scheme': 'SCT'
            },
            raw_message=raw_message
        )
    
    def create_sct_inst(
        self,
        debtor_iban: str,
        creditor_iban: str,
        amount: Decimal,
        reference: str
    ) -> PaymentMessage:
        """Create SEPA Instant Credit Transfer (SCT Inst)"""
        
        if amount > Decimal('100000'):
            raise ValueError("SCT Inst limited to EUR 100,000")
        
        # Similar to SCT but with instant processing flag
        message = self.create_sct(
            debtor_iban=debtor_iban,
            debtor_name="",  # Would be filled from account lookup
            creditor_iban=creditor_iban,
            creditor_name="",  # Would be filled from account lookup
            amount=amount,
            reference=reference
        )
        
        message.metadata['scheme'] = 'SCT_INST'
        message.metadata['max_execution_time'] = 10  # seconds
        
        return message


class ACHProtocol:
    """ACH (Automated Clearing House) protocol implementation"""
    
    def __init__(self):
        self.supported_sec_codes = {
            'PPD': 'Prearranged Payment and Deposit',
            'CCD': 'Corporate Credit or Debit',
            'WEB': 'Internet-Initiated Entry',
            'TEL': 'Telephone-Initiated Entry',
            'CTX': 'Corporate Trade Exchange'
        }
        
    def validate_routing_number(self, routing: str) -> bool:
        """Validate US routing number (ABA)"""
        if len(routing) != 9 or not routing.isdigit():
            return False
        
        # Checksum validation
        checksum = (
            3 * (int(routing[0]) + int(routing[3]) + int(routing[6])) +
            7 * (int(routing[1]) + int(routing[4]) + int(routing[7])) +
            (int(routing[2]) + int(routing[5]) + int(routing[8]))
        )
        
        return checksum % 10 == 0
    
    def create_ach_credit(
        self,
        originator_routing: str,
        originator_account: str,
        originator_name: str,
        receiver_routing: str,
        receiver_account: str,
        receiver_name: str,
        amount: Decimal,
        sec_code: str = 'PPD',
        reference: str = ""
    ) -> PaymentMessage:
        """Create ACH credit transfer"""
        
        if not self.validate_routing_number(originator_routing):
            raise ValueError(f"Invalid originator routing: {originator_routing}")
        if not self.validate_routing_number(receiver_routing):
            raise ValueError(f"Invalid receiver routing: {receiver_routing}")
        if sec_code not in self.supported_sec_codes:
            raise ValueError(f"Invalid SEC code: {sec_code}")
        
        message_id = f"ACH-{uuid4().hex[:16].upper()}"
        
        # Build NACHA format entry
        entry = {
            'record_type': '6',  # Entry Detail Record
            'transaction_code': '22',  # Credit to checking account
            'receiving_dfi': receiver_routing[:8],
            'check_digit': receiver_routing[8],
            'receiving_account': receiver_account[:17].ljust(17),
            'amount': str(int(amount * 100)).zfill(10),  # In cents
            'individual_id': reference[:15].ljust(15),
            'individual_name': receiver_name[:22].ljust(22),
            'discretionary_data': '  ',
            'addenda_indicator': '0',
            'trace_number': f"{originator_routing[:8]}{message_id[-7:]}"
        }
        
        raw_message = ''.join(entry.values())
        
        return PaymentMessage(
            id=message_id,
            network=PaymentNetwork.ACH,
            message_type=MessageType.PAIN001,  # Using ISO 20022 equivalent
            sender=f"{originator_routing}:{originator_account}",
            receiver=f"{receiver_routing}:{receiver_account}",
            amount=amount,
            currency='USD',
            value_date=datetime.now(timezone.utc),
            reference=reference,
            metadata={
                'originator_name': originator_name,
                'receiver_name': receiver_name,
                'sec_code': sec_code,
                'transaction_code': '22',
                'entry_class': self.supported_sec_codes[sec_code]
            },
            raw_message=raw_message
        )


class FedWireProtocol:
    """FedWire payment protocol implementation"""
    
    def __init__(self):
        self.message_types = {
            '1000': 'Customer Transfer Plus',
            '1500': 'Customer Transfer',
            '1520': 'Basic Customer Transfer'
        }
    
    def create_fedwire_transfer(
        self,
        sender_aba: str,
        sender_name: str,
        receiver_aba: str,
        receiver_name: str,
        amount: Decimal,
        reference: str,
        purpose: str = ""
    ) -> PaymentMessage:
        """Create FedWire transfer message"""
        
        message_id = f"FW-{uuid4().hex[:16].upper()}"
        imad = f"{datetime.now().strftime('%Y%m%d')}{message_id[-8:]}"
        
        # Build FedWire message format
        fedwire_fields = {
            '{1100}': '1000',  # Message Type - Customer Transfer Plus
            '{1110}': imad,  # IMAD (Input Message Accountability Data)
            '{1120}': datetime.now().strftime('%Y%m%d%H%M'),  # Sender Supplied Info
            '{1500}': '1000',  # Type/Subtype
            '{1510}': imad,  # IMAD
            '{1520}': str(int(amount * 100)),  # Amount in cents
            '{2000}': f"{sender_aba}{sender_name[:35]}",  # Sender FI
            '{3100}': f"{receiver_aba}{receiver_name[:35]}",  # Receiver FI
            '{3320}': reference[:16],  # Reference for Beneficiary
            '{3500}': purpose[:140] if purpose else 'PAYMENT',  # Originator to Beneficiary Info
        }
        
        raw_message = ''.join(f"{k}{v}" for k, v in fedwire_fields.items())
        
        return PaymentMessage(
            id=message_id,
            network=PaymentNetwork.FEDWIRE,
            message_type=MessageType.PACS008,  # Using ISO 20022 equivalent
            sender=sender_aba,
            receiver=receiver_aba,
            amount=amount,
            currency='USD',
            value_date=datetime.now(timezone.utc),
            reference=reference,
            metadata={
                'sender_name': sender_name,
                'receiver_name': receiver_name,
                'imad': imad,
                'message_type': '1000',
                'purpose': purpose
            },
            raw_message=raw_message
        )


class PaymentRouter:
    """Intelligent payment routing engine"""
    
    def __init__(self):
        self.swift = SWIFTProtocol()
        self.sepa = SEPAProtocol()
        self.ach = ACHProtocol()
        self.fedwire = FedWireProtocol()
        self.routing_rules = self._initialize_routing_rules()
        
    def _initialize_routing_rules(self) -> Dict[str, Any]:
        """Initialize payment routing rules"""
        return {
            'amount_thresholds': {
                'USD': {
                    'ACH': Decimal('1000000'),  # ACH for amounts up to $1M
                    'FEDWIRE': Decimal('999999999')  # FedWire for larger amounts
                },
                'EUR': {
                    'SEPA': Decimal('999999999'),  # SEPA for EUR transfers
                    'SEPA_INST': Decimal('100000')  # SEPA Instant up to €100k
                }
            },
            'speed_priority': {
                'instant': ['SEPA_INST', 'FEDWIRE', 'FASTER_PAYMENTS'],
                'same_day': ['FEDWIRE', 'ACH_SAME_DAY', 'SEPA'],
                'next_day': ['ACH', 'SEPA'],
                'standard': ['SWIFT', 'ACH', 'SEPA']
            },
            'cross_border': ['SWIFT', 'FEDWIRE'],
            'domestic': {
                'US': ['ACH', 'FEDWIRE'],
                'EU': ['SEPA', 'SEPA_INST'],
                'GB': ['FASTER_PAYMENTS', 'SWIFT']
            }
        }
    
    def determine_optimal_route(
        self,
        sender_country: str,
        receiver_country: str,
        amount: Decimal,
        currency: str,
        speed: str = 'standard'
    ) -> PaymentNetwork:
        """Determine optimal payment route based on parameters"""
        
        # Cross-border payments
        if sender_country != receiver_country:
            if currency == 'EUR' and sender_country in self.sepa.sepa_countries and receiver_country in self.sepa.sepa_countries:
                if amount <= Decimal('100000') and speed == 'instant':
                    return PaymentNetwork.SEPA  # Would use SEPA Instant
                return PaymentNetwork.SEPA
            return PaymentNetwork.SWIFT
        
        # Domestic payments
        if sender_country == 'US' and currency == 'USD':
            if amount > Decimal('1000000') or speed == 'instant':
                return PaymentNetwork.FEDWIRE
            return PaymentNetwork.ACH
        
        if sender_country in self.sepa.sepa_countries and currency == 'EUR':
            if amount <= Decimal('100000') and speed == 'instant':
                return PaymentNetwork.SEPA  # Would use SEPA Instant
            return PaymentNetwork.SEPA
        
        # Default to SWIFT for other cases
        return PaymentNetwork.SWIFT
    
    async def route_payment(
        self,
        sender_details: Dict[str, Any],
        receiver_details: Dict[str, Any],
        amount: Decimal,
        currency: str,
        reference: str,
        speed: str = 'standard'
    ) -> PaymentMessage:
        """Route payment through optimal network"""
        
        # Determine routing
        network = self.determine_optimal_route(
            sender_details.get('country', ''),
            receiver_details.get('country', ''),
            amount,
            currency,
            speed
        )
        
        # Create appropriate message based on network
        if network == PaymentNetwork.SWIFT:
            return self.swift.create_mt103(
                sender_bic=sender_details.get('bic', ''),
                receiver_bic=receiver_details.get('bic', ''),
                sender_account=sender_details.get('account', ''),
                receiver_account=receiver_details.get('account', ''),
                amount=amount,
                currency=currency,
                reference=reference
            )
        
        elif network == PaymentNetwork.SEPA:
            return self.sepa.create_sct(
                debtor_iban=sender_details.get('iban', ''),
                debtor_name=sender_details.get('name', ''),
                creditor_iban=receiver_details.get('iban', ''),
                creditor_name=receiver_details.get('name', ''),
                amount=amount,
                reference=reference
            )
        
        elif network == PaymentNetwork.ACH:
            return self.ach.create_ach_credit(
                originator_routing=sender_details.get('routing', ''),
                originator_account=sender_details.get('account', ''),
                originator_name=sender_details.get('name', ''),
                receiver_routing=receiver_details.get('routing', ''),
                receiver_account=receiver_details.get('account', ''),
                receiver_name=receiver_details.get('name', ''),
                amount=amount,
                reference=reference
            )
        
        elif network == PaymentNetwork.FEDWIRE:
            return self.fedwire.create_fedwire_transfer(
                sender_aba=sender_details.get('routing', ''),
                sender_name=sender_details.get('name', ''),
                receiver_aba=receiver_details.get('routing', ''),
                receiver_name=receiver_details.get('name', ''),
                amount=amount,
                reference=reference
            )
        
        else:
            raise ValueError(f"Unsupported payment network: {network}")


class PaymentGateway:
    """Main payment gateway orchestrator"""
    
    def __init__(self):
        self.router = PaymentRouter()
        self.message_queue = asyncio.Queue()
        self.network_connections = {}
        
    async def initialize(self):
        """Initialize payment gateway connections"""
        # In production, would establish connections to payment networks
        pass
    
    async def send_payment(
        self,
        sender: Dict[str, Any],
        receiver: Dict[str, Any],
        amount: Decimal,
        currency: str,
        reference: str,
        speed: str = 'standard'
    ) -> Tuple[bool, str]:
        """Send payment through appropriate network"""
        
        try:
            # Route payment
            message = await self.router.route_payment(
                sender,
                receiver,
                amount,
                currency,
                reference,
                speed
            )
            
            # Queue for transmission
            await self.message_queue.put(message)
            
            # In production, would actually transmit to network
            # For now, simulate successful transmission
            await asyncio.sleep(0.1)
            
            return True, message.id
            
        except Exception as e:
            return False, str(e)
    
    async def get_payment_status(self, payment_id: str) -> Dict[str, Any]:
        """Get payment status"""
        # In production, would query payment network
        return {
            'id': payment_id,
            'status': 'COMPLETED',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# Example usage
async def main():
    """Test payment protocols"""
    
    # Initialize gateway
    gateway = PaymentGateway()
    await gateway.initialize()
    
    # Test SWIFT payment
    sender = {
        'country': 'US',
        'bic': 'CHASUS33XXX',
        'account': '123456789',
        'name': 'John Doe'
    }
    
    receiver = {
        'country': 'GB',
        'bic': 'BARCGB22XXX',
        'account': '987654321',
        'name': 'Jane Smith'
    }
    
    success, payment_id = await gateway.send_payment(
        sender,
        receiver,
        Decimal('10000.00'),
        'USD',
        'REF-' + uuid4().hex[:8].upper(),
        'standard'
    )
    
    print(f"Payment sent: {success}, ID: {payment_id}")
    
    # Test SEPA payment
    sender_sepa = {
        'country': 'DE',
        'iban': 'DE89370400440532013000',
        'name': 'Max Mustermann'
    }
    
    receiver_sepa = {
        'country': 'FR',
        'iban': 'FR1420041010050500013M02606',
        'name': 'Pierre Dupont'
    }
    
    success, payment_id = await gateway.send_payment(
        sender_sepa,
        receiver_sepa,
        Decimal('500.00'),
        'EUR',
        'SEPA-' + uuid4().hex[:8].upper(),
        'instant'
    )
    
    print(f"SEPA payment sent: {success}, ID: {payment_id}")


if __name__ == "__main__":
    asyncio.run(main())