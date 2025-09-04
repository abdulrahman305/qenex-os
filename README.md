# Unified System

A complete financial system implementation with authentication, tokens, and DeFi capabilities.

## 🚀 Quick Start

```bash
# Install and run
pip install -r requirements.txt
python system.py
```

## 📋 Features

### Core Components
- **Authentication System** - Secure user management with session handling
- **Token System** - Create and manage digital tokens
- **DeFi Platform** - Automated market maker with liquidity pools
- **Monitoring** - Real-time system metrics and health checks

### Technical Specifications
- **Database**: SQLite with thread-safe operations
- **Security**: PBKDF2 password hashing (200k iterations)
- **Precision**: 28-digit decimal accuracy
- **Performance**: Connection pooling and caching

## 📊 System Architecture

```
┌─────────────────────────────────────┐
│         Unified System              │
├─────────────────────────────────────┤
│  Auth │ Tokens │ DeFi │ Monitor    │
├───────┴────────┴──────┴─────────────┤
│         Database Layer              │
└─────────────────────────────────────┘
```

## 📖 Documentation

- [Architecture Overview](./architecture.md) - Technical system design
- [Visual Guide](./visual_guide.md) - Interactive diagrams and flows
- [API Reference](./docs/api.md) - Endpoint documentation

## 💻 Usage Examples

### User Management
```python
from system import UnifiedSystem

system = UnifiedSystem()

# Register user
user_id = system.auth.register("username", "password")

# Login
session = system.auth.login("username", "password")
```

### Token Operations
```python
# Create token
system.tokens.create_token("BTC", "Bitcoin", decimals=8)

# Transfer tokens
tx_id = system.tokens.transfer(from_user, to_user, "BTC", amount)

# Check balance
balance = system.tokens.get_balance(user_id, "BTC")
```

### DeFi Operations
```python
# Create liquidity pool
pool_id = system.defi.create_pool("ETH", "USDC")

# Add liquidity
shares = system.defi.add_liquidity(user_id, "ETH", "USDC", 
                                   amount_eth, amount_usdc)

# Swap tokens
output = system.defi.swap(user_id, "ETH", "USDC", input_amount)
```

## 🔒 Security Features

- **Password Security**: PBKDF2-SHA256 with salt
- **Session Management**: Secure token generation
- **SQL Injection Prevention**: Parameterized queries
- **Audit Logging**: Complete action trail
- **Input Validation**: Type and range checking

## 📈 Performance

- Thread-safe database operations
- Connection pooling for scalability
- In-memory session caching
- Automatic cleanup routines
- Metric collection and aggregation

## 🗂 Database Schema

### Core Tables
- `users` - User accounts
- `sessions` - Active sessions
- `tokens` - Token definitions
- `balances` - User holdings
- `transactions` - Transaction history
- `pools` - Liquidity pools
- `liquidity` - LP positions
- `metrics` - System metrics
- `audit_log` - Security audit

## 🧪 Testing

Run the built-in demonstration:
```bash
python system.py
```

This will:
1. Create a demo user
2. Authenticate the user
3. Mint tokens
4. Add liquidity to pools
5. Perform token swaps
6. Display system dashboard

## 📦 Requirements

- Python 3.8+
- SQLite3
- See `requirements.txt` for Python packages

## 🛠 Configuration

Edit configuration in `system.py`:
```python
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
LOG_DIR = BASE_DIR / 'logs'
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines first.

## 📞 Support

For issues and questions, please use GitHub Issues.

---

**Note**: This is a demonstration system. For production use, additional security hardening and infrastructure components are recommended.