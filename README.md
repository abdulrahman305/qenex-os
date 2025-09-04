# QENEX Operating System

A unified financial system with token management and automated market making.

## 🚀 Quick Start

```bash
# Install and run
pip install -r requirements.txt
python main.py
```

## 📁 Project Structure

```
qenex-os/
├── main.py                 # Core system implementation
├── docs/
│   ├── QUICK_START.md     # Getting started guide
│   └── SYSTEM_OVERVIEW.md # Architecture documentation
├── data/                  # Database storage
├── logs/                  # System logs
└── requirements.txt       # Dependencies
```

## 🏗 Architecture

```
┌─────────────────────────────────────────────┐
│              MAIN SYSTEM                    │
├─────────────────────────────────────────────┤
│   Database │ Tokens │ Pools │ Accounts     │
├────────────┴────────┴───────┴───────────────┤
│            SQLite Database                  │
└─────────────────────────────────────────────┘
```

## ⚙️ Core Features

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