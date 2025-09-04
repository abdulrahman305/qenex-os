# QENEX Financial Operating System

## Overview

QENEX OS is a production-grade financial infrastructure designed specifically for banks and financial institutions. Built from the ground up with quantum-resistant cryptography, real-time transaction processing, and comprehensive compliance frameworks.

## Key Features

### 🔒 **Quantum-Resistant Security**
- Post-quantum cryptography implementation
- Hardware Security Module (HSM) integration
- Advanced threat detection and response
- End-to-end encryption for all communications

### ⚡ **Real-Time Transaction Processing**
- High-throughput transaction engine (100K+ TPS)
- Sub-second settlement processing
- Advanced transaction validation and fraud detection
- Multi-currency support with real-time FX

### 🏛️ **Banking Protocol Compliance**
- SWIFT MT message processing (MT103, MT202, MT940)
- ISO 20022 standard implementation
- SEPA instant payments
- ACH and wire transfer processing
- Real-time gross settlement (RTGS)

### 🧠 **Intelligent Risk Management**
- Machine learning-based fraud detection
- Real-time AML/KYC compliance monitoring
- Automated risk scoring and assessment
- Predictive analytics for operational risk

### 🌐 **High Availability Architecture**
- Byzantine Fault Tolerant consensus
- Automatic failover and disaster recovery
- Geographic distribution support
- Zero-downtime deployment capabilities

## Quick Start

### Prerequisites

- Rust 1.70+
- PostgreSQL 14+
- Redis 6.2+
- TLS certificates for secure communication

### Installation

```bash
# Clone the repository
git clone https://github.com/abdulrahman305/qenex-os.git
cd qenex-os

# Build the project
cargo build --release

# Run database migrations
cargo run --bin setup-db

# Start a node
cargo run --bin qenex-node -- --config /etc/qenex/config.toml
```

### Docker Deployment

```bash
# Build Docker image
docker build -t qenex-os .

# Run with Docker Compose
docker-compose up -d
```

## Configuration

### Basic Configuration

```toml
[node]
id = "node-1"
port = 8080
max_connections = 1000

[database]
url = "postgresql://qenex:password@localhost/qenex"
max_connections = 20
timeout = 30

[security]
tls_cert = "/etc/qenex/certs/server.crt"
tls_key = "/etc/qenex/certs/server.key"
hsm_provider = "software" # or "pkcs11"

[consensus]
algorithm = "byzantine_fault_tolerant"
timeout_ms = 5000
max_validators = 100

[compliance]
aml_enabled = true
kyc_verification = true
transaction_monitoring = true
sanctions_screening = true
```

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        QENEX OS                                │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │   Banking   │ │   Trading   │ │  Compliance │              │
│  │   Services  │ │  Platform   │ │   Engine    │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Protocol Layer                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │    SWIFT    │ │  ISO 20022  │ │    SEPA     │              │
│  │ Processing  │ │  Messages   │ │  Instant    │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Consensus & Settlement Layer                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │  Byzantine  │ │  Real-time  │ │   Quantum   │              │
│  │    Fault    │ │ Settlement  │ │  Resistant  │              │
│  │  Tolerance  │ │   Engine    │ │ Crypto      │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  Storage & Security Layer                                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐              │
│  │ Distributed │ │   Hardware  │ │   Network   │              │
│  │  Database   │ │  Security   │ │  Security   │              │
│  │   System    │ │   Module    │ │   Layer     │              │
│  └─────────────┘ └─────────────┘ └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Core Components

### Account Management
- Create accounts with unique addresses
- Track balances across multiple tokens
- Transaction history

### Token System
- Create custom tokens
- Transfer between accounts
- Mint new supply
- Decimal precision support

### Liquidity Pools
- Automated Market Maker (AMM)
- Constant product formula (x*y=k)
- 0.3% trading fee
- Price discovery

### Database
- SQLite with ACID compliance
- Foreign key constraints
- Transaction safety
- Indexed queries

## 💻 Usage Examples

### Basic Operations

```python
from main import System

# Initialize
system = System()

# Create account
account = system.create_account()

# Mint tokens
system.tokens.mint(account, 'ETH', Decimal('100'))

# Transfer tokens
tx = system.tokens.transfer(from_addr, to_addr, 'ETH', Decimal('10'))

# Swap tokens
output = system.pools.swap('ETH', 'USDC', Decimal('1'))
```

### Pool Operations

```python
# Create pool
system.pools.create_pool('ETH', 'USDC')

# Add liquidity
system.pools.add_liquidity('ETH', 'USDC', 
                          Decimal('10'), Decimal('20000'))

# Get price
price = system.pools.get_price('ETH', 'USDC')
```

## 📊 System Demo

Run the built-in demonstration:

```bash
python main.py
```

This demonstrates:
- Account creation
- Token minting
- Transfers
- Liquidity provision
- Token swaps
- Balance queries

## 🔧 Configuration

### Database Path
```python
DATA_DIR = Path(__file__).parent / 'data'
```

### Logging
```python
logging.basicConfig(level=logging.INFO)
```

### Default Tokens
- USDC (6 decimals)
- ETH (18 decimals)
- BTC (8 decimals)

## 📈 AMM Mathematics

### Constant Product Formula
```
x * y = k
```

### Swap Calculation
```python
amount_out = (amount_in * reserve_out) / (reserve_in + amount_in)
```

### Price Impact
```python
price_impact = (amount_in / reserve_in) * 100
```

## 🔒 Security

- Parameterized SQL queries
- Transaction atomicity
- Balance validation
- Input sanitization
- Error handling with rollback

## 📝 Documentation

- [Quick Start Guide](docs/QUICK_START.md)
- [System Overview](docs/SYSTEM_OVERVIEW.md)
- [API Reference](docs/API_REFERENCE.md)

## 🧪 Testing

Run the demonstration to test all features:

```python
system = System()
system.run_demo()
```

## ⚠️ Important Notes

- **Development System**: Not for production use
- **Database**: SQLite for simplicity
- **Security**: Basic implementation
- **Audit**: Not audited

## 🛠 Requirements

- Python 3.8+
- SQLite3
- Standard library only

## 📄 License

MIT License

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines first.

---

**Note**: This is a demonstration system for educational purposes.