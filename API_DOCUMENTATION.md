# QENEX Banking System - API Documentation

## Overview

The QENEX Banking System provides comprehensive APIs for banking operations, AI/ML fraud detection, smart contract deployment, and cross-platform compatibility. This documentation covers all available endpoints and integration methods.

## Table of Contents

1. [Authentication](#authentication)
2. [Core Banking API](#core-banking-api)
3. [AI/ML API](#aiml-api)
4. [Banking Protocols API](#banking-protocols-api)
5. [Smart Contract API](#smart-contract-api)
6. [System Management API](#system-management-api)
7. [Kernel Module Interface](#kernel-module-interface)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)
10. [Examples](#examples)

## Authentication

All API endpoints require authentication using JWT tokens or API keys.

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "user123",
  "password": "SecurePass123!",
  "mfa_token": "123456"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 28800,
  "user_id": "user_12345",
  "permissions": ["banking:read", "banking:write"]
}
```

### Token Refresh
```http
POST /auth/refresh
Authorization: Bearer <token>
```

## Core Banking API

### Account Management

#### Create Account
```http
POST /api/v1/accounts
Authorization: Bearer <token>
Content-Type: application/json

{
  "account_type": "CHECKING",
  "currency": "USD", 
  "initial_balance": 1000.00,
  "account_holder": "John Doe"
}
```

**Response:**
```json
{
  "account_number": "ACC_1234567890",
  "account_type": "CHECKING",
  "currency": "USD",
  "balance": 1000.00,
  "created_at": "2024-01-15T10:30:00Z",
  "status": "ACTIVE"
}
```

#### Get Account Details
```http
GET /api/v1/accounts/{account_number}
Authorization: Bearer <token>
```

#### Get Account Balance
```http
GET /api/v1/accounts/{account_number}/balance
Authorization: Bearer <token>
```

**Response:**
```json
{
  "account_number": "ACC_1234567890",
  "balance": 1000.00,
  "available_balance": 950.00,
  "currency": "USD",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

#### List Accounts
```http
GET /api/v1/accounts?limit=20&offset=0
Authorization: Bearer <token>
```

### Transaction Processing

#### Process Transaction
```http
POST /api/v1/transactions
Authorization: Bearer <token>
Content-Type: application/json

{
  "from_account": "ACC_1234567890",
  "to_account": "ACC_0987654321", 
  "amount": 100.00,
  "currency": "USD",
  "description": "Payment for services",
  "reference": "INV_001",
  "metadata": {
    "merchant_id": "MERCH_123",
    "category": "services"
  }
}
```

**Response:**
```json
{
  "transaction_id": "TX_1234567890123456",
  "status": "COMPLETED",
  "from_account": "ACC_1234567890",
  "to_account": "ACC_0987654321",
  "amount": 100.00,
  "currency": "USD",
  "description": "Payment for services",
  "processed_at": "2024-01-15T10:30:00Z",
  "confirmation_number": "CONF_ABC123",
  "fraud_score": 0.05,
  "fees": {
    "processing_fee": 0.50,
    "total_fees": 0.50
  }
}
```

#### Get Transaction Details
```http
GET /api/v1/transactions/{transaction_id}
Authorization: Bearer <token>
```

#### Reverse Transaction
```http
POST /api/v1/transactions/{transaction_id}/reverse
Authorization: Bearer <token>
Content-Type: application/json

{
  "reason": "Fraudulent transaction",
  "reversal_type": "FULL"
}
```

#### List Transactions
```http
GET /api/v1/transactions?account={account_number}&limit=50&offset=0&start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer <token>
```

### Ledger Operations

#### Get Ledger Entries
```http
GET /api/v1/ledger?account={account_number}&limit=100
Authorization: Bearer <token>
```

**Response:**
```json
{
  "entries": [
    {
      "entry_id": "LED_123456789",
      "account_number": "ACC_1234567890",
      "transaction_id": "TX_1234567890123456",
      "entry_type": "DEBIT",
      "amount": 100.00,
      "balance_after": 900.00,
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total_count": 1,
  "pagination": {
    "limit": 100,
    "offset": 0,
    "has_more": false
  }
}
```

## AI/ML API

### Fraud Detection

#### Analyze Transaction
```http
POST /api/v1/ai/fraud/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "transaction_id": "TX_1234567890123456",
  "user_id": "user_123",
  "amount": 500.00,
  "merchant": "online_electronics",
  "location": "US",
  "timestamp": "2024-01-15T10:30:00Z",
  "payment_method": "credit_card"
}
```

**Response:**
```json
{
  "transaction_id": "TX_1234567890123456",
  "fraud_probability": 0.15,
  "risk_level": "LOW",
  "action": "APPROVE",
  "risk_factors": [
    {
      "factor": "unusual_amount",
      "score": 0.2,
      "description": "Amount is 50% higher than user average"
    },
    {
      "factor": "merchant_risk",
      "score": 0.1,
      "description": "Electronics merchant has low risk profile"
    }
  ],
  "model_version": "fraud_detection_v2.1",
  "processing_time_ms": 45
}
```

#### Get Fraud Model Metrics
```http
GET /api/v1/ai/fraud/metrics
Authorization: Bearer <token>
```

### Risk Assessment

#### Assess Credit Risk
```http
POST /api/v1/ai/risk/credit
Authorization: Bearer <token>
Content-Type: application/json

{
  "customer_id": "CUST_123456",
  "age": 35,
  "annual_income": 75000,
  "employment_years": 8,
  "credit_history_length": 10,
  "num_credit_accounts": 3,
  "credit_utilization": 0.25,
  "payment_history_score": 0.95,
  "debt_to_income": 0.20
}
```

**Response:**
```json
{
  "customer_id": "CUST_123456",
  "risk_score": 0.15,
  "risk_category": "LOW",
  "recommended_credit_limit": 25000.00,
  "interest_rate": 0.08,
  "approval_probability": 0.92,
  "risk_factors": {
    "positive": ["high_income", "stable_employment", "low_utilization"],
    "negative": ["short_credit_history"]
  },
  "assessment_date": "2024-01-15T10:30:00Z"
}
```

#### Train Model
```http
POST /api/v1/ai/models/train
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_type": "fraud_detection",
  "training_data_period": "30_days",
  "parameters": {
    "n_estimators": 100,
    "max_depth": 10
  }
}
```

## Banking Protocols API

### ISO 20022

#### Generate Payment Message
```http
POST /api/v1/protocols/iso20022/pain001
Authorization: Bearer <token>
Content-Type: application/json

{
  "message_id": "MSG_123456789",
  "initiating_party": {
    "name": "ABC Corporation"
  },
  "payment_info": [
    {
      "payment_id": "PMT_001",
      "amount": 1000.50,
      "currency": "EUR",
      "debtor_name": "John Doe",
      "debtor_iban": "DE89370400440532013000",
      "creditor_name": "Jane Smith", 
      "creditor_iban": "FR1420041010050500013M02606",
      "remittance_info": "Invoice payment #12345"
    }
  ]
}
```

**Response:**
```json
{
  "message_id": "MSG_123456789",
  "message_type": "pain.001.001.09",
  "xml_content": "<?xml version=\"1.0\" encoding=\"UTF-8\"?>...",
  "validation_status": "VALID",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Parse ISO 20022 Message
```http
POST /api/v1/protocols/iso20022/parse
Authorization: Bearer <token>
Content-Type: application/xml

<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pain.001.001.09">
  <!-- ISO 20022 XML content -->
</Document>
```

### SWIFT

#### Generate MT103 Message
```http
POST /api/v1/protocols/swift/mt103
Authorization: Bearer <token>
Content-Type: application/json

{
  "sender": "DEUTDEFF",
  "receiver": "BNPAFRPP", 
  "reference": "REF123456789",
  "amount": "1000,50",
  "currency": "EUR",
  "value_date": "240115",
  "ordering_customer": "JOHN DOE\nACCOUNT 123456789",
  "beneficiary_customer": "/FR1420041010050500013M02606\nJANE SMITH",
  "remittance_info": "INVOICE PAYMENT 12345"
}
```

**Response:**
```json
{
  "message_type": "MT103",
  "swift_message": "{1:F01DEUTDEFF0000000000}{2:I103BNPAFRPPN}{4:\n:20:REF123456789\n:23B:CRED\n:32A:240115EUR1000,50\n:50K:JOHN DOE\nACCOUNT 123456789\n:59:/FR1420041010050500013M02606\nJANE SMITH\n:70:INVOICE PAYMENT 12345\n:71A:SHA\n-}",
  "validation_status": "VALID",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### SEPA

#### Process SEPA Credit Transfer
```http
POST /api/v1/protocols/sepa/sct
Authorization: Bearer <token>
Content-Type: application/json

{
  "payment_id": "SEPA_PMT_001",
  "amount": 250.00,
  "currency": "EUR",
  "debtor_name": "Alice Johnson",
  "debtor_iban": "DE89370400440532013000",
  "debtor_bic": "DEUTDEFF",
  "creditor_name": "Bob Wilson", 
  "creditor_iban": "FR1420041010050500013M02606",
  "creditor_bic": "BNPAFRPP",
  "remittance_info": "Monthly rent payment",
  "execution_date": "2024-01-16"
}
```

**Response:**
```json
{
  "payment_id": "SEPA_PMT_001",
  "transaction_type": "SCT",
  "status": "PROCESSED",
  "xml_content": "<?xml version=\"1.0\"?>...",
  "processing_time": "2024-01-15T10:30:00Z",
  "settlement_date": "2024-01-16",
  "fees": {
    "processing_fee": 0.00,
    "total_fees": 0.00
  }
}
```

## Smart Contract API

### Contract Deployment

#### Deploy Payment Contract
```http
POST /api/v1/contracts/deploy
Authorization: Bearer <token>
Content-Type: application/json

{
  "contract_type": "PAYMENT_PROCESSING",
  "network": "ethereum",
  "constructor_args": [],
  "gas_limit": 3000000
}
```

**Response:**
```json
{
  "contract_id": "CONTRACT_1234567890",
  "contract_type": "PAYMENT_PROCESSING",
  "network": "ethereum",
  "address": "0x742f35Cc6584C2c4b4Dda2a8b8e1A8F5b8C2F",
  "transaction_hash": "0x9fc76417374aa880d4449a1f7f31ec597f00b1f6f3dd2d06f4df9c7bae5ab22",
  "deployment_time": "2024-01-15T10:30:00Z",
  "gas_used": 2847392,
  "status": "deployed",
  "verification_status": "pending"
}
```

#### Execute Contract Function
```http
POST /api/v1/contracts/{contract_id}/execute
Authorization: Bearer <token>
Content-Type: application/json

{
  "function_name": "initiatePayment",
  "parameters": [
    "0x8ba1f109551bd432803012645hac136c0c8b48fc",
    1000000000000000000,
    "0x1234567890123456789012345678901234567890"
  ],
  "gas_limit": 100000
}
```

### Contract Management

#### List Deployed Contracts
```http
GET /api/v1/contracts?network=ethereum&limit=20
Authorization: Bearer <token>
```

#### Get Contract Details
```http
GET /api/v1/contracts/{contract_id}
Authorization: Bearer <token>
```

#### Verify Contract
```http
POST /api/v1/contracts/{contract_id}/verify
Authorization: Bearer <token>
```

## System Management API

### System Status

#### Get System Health
```http
GET /api/v1/system/health
Authorization: Bearer <token>
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 86400,
  "components": {
    "database": "healthy",
    "ai_ml_system": "healthy", 
    "blockchain": "healthy",
    "protocols": "healthy",
    "smart_contracts": "healthy"
  },
  "metrics": {
    "total_transactions": 1543821,
    "transactions_per_second": 145.2,
    "average_response_time_ms": 12.5,
    "error_rate": 0.001,
    "active_sessions": 892
  },
  "platform_info": {
    "os": "Linux",
    "architecture": "x86_64",
    "memory_usage_mb": 384.2,
    "cpu_usage_percent": 15.8
  }
}
```

#### Get System Metrics
```http
GET /api/v1/system/metrics?period=24h
Authorization: Bearer <token>
```

#### Get Performance Statistics
```http
GET /api/v1/system/performance
Authorization: Bearer <token>
```

### Configuration Management

#### Get System Configuration
```http
GET /api/v1/system/config
Authorization: Bearer <token>
```

#### Update Configuration
```http
PATCH /api/v1/system/config
Authorization: Bearer <token>
Content-Type: application/json

{
  "max_concurrent_transactions": 2000,
  "enable_ai_ml": true,
  "security_level": "HIGH"
}
```

## Kernel Module Interface

For Linux systems, QENEX provides a kernel module interface for high-performance operations.

### Device Access
```c
#include <fcntl.h>
#include <sys/ioctl.h>
#include "qenex_ioctl.h"

int fd = open("/dev/qenex_banking", O_RDWR);
```

### IOCTL Commands

#### Create Account
```c
struct qenex_account_request req = {
    .initial_balance = 100000,  // Amount in cents
    .currency = 840,           // USD currency code
    .account_type = 1          // CHECKING account
};

int result = ioctl(fd, QENEX_CREATE_ACCOUNT, &req);
// req.account_number will contain the new account number
```

#### Transfer Funds
```c
struct qenex_transfer_request req = {
    .from_account = 1234567890,
    .to_account = 987654321,
    .amount = 10000,           // $100.00 in cents
    .currency = 840            // USD
};

int result = ioctl(fd, QENEX_TRANSFER, &req);
// req.transaction_id will contain the transaction ID
```

#### Get Balance
```c
struct qenex_balance_request req = {
    .account_number = 1234567890
};

int result = ioctl(fd, QENEX_GET_BALANCE, &req);
// req.balance will contain current balance in cents
```

### Python Kernel Interface
```python
import os
import struct
import fcntl

# Open device
fd = os.open('/dev/qenex_banking', os.O_RDWR)

# Create account (simplified example)
account_data = struct.pack('III', 100000, 840, 1)  # balance, currency, type
result = fcntl.ioctl(fd, 0x1001, account_data)

# Close device
os.close(fd)
```

## Error Handling

### Standard Error Response Format
```json
{
  "error": {
    "code": "INSUFFICIENT_FUNDS",
    "message": "Account balance insufficient for transaction",
    "details": {
      "account_balance": 50.00,
      "requested_amount": 100.00,
      "required_amount": 100.50
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "request_id": "req_1234567890"
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_ACCOUNT` | 400 | Account number not found or invalid |
| `INSUFFICIENT_FUNDS` | 400 | Account balance too low |
| `TRANSACTION_LIMIT_EXCEEDED` | 400 | Transaction exceeds daily/monthly limits |
| `FRAUD_DETECTED` | 403 | Transaction blocked by fraud detection |
| `UNAUTHORIZED` | 401 | Authentication required or invalid |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `SYSTEM_MAINTENANCE` | 503 | System temporarily unavailable |
| `PROTOCOL_ERROR` | 400 | Banking protocol validation failed |
| `CONTRACT_EXECUTION_FAILED` | 500 | Smart contract execution error |

### Error Handling Best Practices

1. **Retry Logic**: Implement exponential backoff for transient errors
2. **Logging**: Log all errors with request context
3. **User Experience**: Provide clear error messages to end users
4. **Monitoring**: Set up alerts for high error rates

## Rate Limiting

### Default Limits

| Endpoint Category | Requests per Minute | Burst Limit |
|------------------|-------------------|-------------|
| Authentication | 10 | 20 |
| Account Operations | 100 | 200 |
| Transaction Processing | 60 | 120 |
| AI/ML Analysis | 1000 | 2000 |
| System Management | 30 | 60 |

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again later.",
    "retry_after": 60
  }
}
```

## Examples

### Complete Transaction Processing Example

```python
import requests
import json
from datetime import datetime

# Configuration
BASE_URL = "https://api.qenex.com"
API_TOKEN = "your_jwt_token_here"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Step 1: Create accounts
def create_account(account_type, currency, initial_balance):
    payload = {
        "account_type": account_type,
        "currency": currency,
        "initial_balance": initial_balance
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/accounts",
        headers=headers,
        json=payload
    )
    
    return response.json()

# Create source and destination accounts
source_account = create_account("CHECKING", "USD", 5000.00)
dest_account = create_account("SAVINGS", "USD", 1000.00)

print(f"Source Account: {source_account['account_number']}")
print(f"Destination Account: {dest_account['account_number']}")

# Step 2: Process transaction with AI fraud detection
def process_transaction(from_account, to_account, amount):
    # First, analyze for fraud
    fraud_payload = {
        "user_id": "user_123",
        "amount": amount,
        "merchant": "internal_transfer",
        "location": "US",
        "timestamp": datetime.now().isoformat(),
        "payment_method": "bank_transfer"
    }
    
    fraud_response = requests.post(
        f"{BASE_URL}/api/v1/ai/fraud/analyze",
        headers=headers,
        json=fraud_payload
    )
    
    fraud_result = fraud_response.json()
    print(f"Fraud Analysis: {fraud_result['risk_level']} risk")
    
    if fraud_result['action'] != 'APPROVE':
        print("Transaction blocked by fraud detection")
        return None
    
    # Process the transaction
    transaction_payload = {
        "from_account": from_account,
        "to_account": to_account,
        "amount": amount,
        "currency": "USD",
        "description": "Savings transfer",
        "reference": "SAVE_001"
    }
    
    transaction_response = requests.post(
        f"{BASE_URL}/api/v1/transactions",
        headers=headers,
        json=transaction_payload
    )
    
    return transaction_response.json()

# Process transfer
transaction_result = process_transaction(
    source_account['account_number'],
    dest_account['account_number'],
    500.00
)

if transaction_result:
    print(f"Transaction completed: {transaction_result['transaction_id']}")
    print(f"Status: {transaction_result['status']}")
    
    # Step 3: Generate SWIFT message for the transaction
    swift_payload = {
        "sender": "QENEXUS33",
        "receiver": "QENEXUS33",
        "reference": transaction_result['transaction_id'],
        "amount": "500,00",
        "currency": "USD",
        "value_date": datetime.now().strftime('%y%m%d'),
        "ordering_customer": "INTERNAL TRANSFER",
        "beneficiary_customer": "SAVINGS ACCOUNT"
    }
    
    swift_response = requests.post(
        f"{BASE_URL}/api/v1/protocols/swift/mt103",
        headers=headers,
        json=swift_payload
    )
    
    swift_result = swift_response.json()
    print(f"SWIFT message generated: {len(swift_result['swift_message'])} characters")
```

### Smart Contract Integration Example

```python
# Deploy a payment processing contract
def deploy_payment_contract():
    payload = {
        "contract_type": "PAYMENT_PROCESSING",
        "network": "localhost",
        "constructor_args": [],
        "gas_limit": 3000000
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/contracts/deploy",
        headers=headers,
        json=payload
    )
    
    return response.json()

# Deploy contract
contract = deploy_payment_contract()
print(f"Contract deployed: {contract['address']}")

# Execute payment through smart contract
def execute_smart_payment(contract_id, receiver, amount):
    payload = {
        "function_name": "initiatePayment",
        "parameters": [receiver, amount, "0x" + "0" * 64],  # reference
        "gas_limit": 100000
    }
    
    response = requests.post(
        f"{BASE_URL}/api/v1/contracts/{contract_id}/execute",
        headers=headers,
        json=payload
    )
    
    return response.json()

# Execute smart contract payment
smart_payment = execute_smart_payment(
    contract['contract_id'],
    "0x742f35Cc6584C2c4b4Dda2a8b8e1A8F5b8C2F123",
    1000000000000000000  # 1 ETH in wei
)

print(f"Smart payment executed: {smart_payment}")
```

This completes the comprehensive API documentation for the QENEX Banking System. The system provides enterprise-grade banking capabilities with modern APIs, real-time fraud detection, and cross-platform compatibility.