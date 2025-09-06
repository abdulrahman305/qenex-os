# Quick Start Guide

## Installation

### Prerequisites
- Python 3.8 or higher
- SQLite3
- pip package manager

### Setup Steps

1. **Clone Repository**
```bash
git clone https://github.com/abdulrahman305/qenex-os.git
cd qenex-os
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run System**
```bash
python main.py
```

## First Run Output

When you run the system for the first time, you'll see:

```
============================================================
 SYSTEM DEMONSTRATION
============================================================

[1] Creating Accounts...
    Account 1: 0x1a2b3c4d...
    Account 2: 0x5e6f7a8b...

[2] Minting Tokens...
    Minted 100 ETH to Account 1
    Minted 200,000 USDC to Account 1
    Minted 10 BTC to Account 2

[3] Account Balances:
    Account 1 - ETH: 100
    Account 1 - USDC: 200000

[4] Transferring Tokens...
    Transfer successful: 0xabcd1234...

[5] Adding Liquidity...
    Added 50 ETH + 100,000 USDC to pool

[6] Pool Information:
    ETH Reserve: 50
    USDC Reserve: 100000
    ETH Price: 2000 USDC

[7] Performing Swap...
    Swapped 1 ETH for 1980.04 USDC

[8] Final Balances:
    Account 1:
      ETH: 89
      USDC: 200000
    Account 2:
      ETH: 10
      BTC: 10

============================================================
 DEMONSTRATION COMPLETE
============================================================
```

## Basic Usage

### Import System
```python
from main import System

system = System()
```

### Create Account
```python
address = system.create_account()
print(f"New account: {address}")
```

### Check Balance
```python
balance = system.tokens.get_balance(address, 'ETH')
print(f"ETH Balance: {balance}")
```

### Transfer Tokens
```python
tx_hash = system.tokens.transfer(
    from_addr=account1,
    to_addr=account2,
    token='ETH',
    amount=Decimal('10')
)
print(f"Transfer TX: {tx_hash}")
```

### Add Liquidity
```python
success = system.pools.add_liquidity(
    token0='ETH',
    token1='USDC',
    amount0=Decimal('10'),
    amount1=Decimal('20000')
)
```

### Swap Tokens
```python
output = system.pools.swap(
    token_in='ETH',
    token_out='USDC',
    amount_in=Decimal('1')
)
print(f"Received: {output} USDC")
```

## File Structure

```
qenex-os/
├── main.py              # Main system entry
├── data/
│   └── main.db         # SQLite database
├── logs/
│   └── system.log      # System logs
├── docs/
│   ├── QUICK_START.md  # This file
│   └── SYSTEM_OVERVIEW.md
└── requirements.txt    # Python dependencies
```

## Common Operations

### View All Tokens
```python
for symbol, info in system.tokens.tokens.items():
    print(f"{symbol}: {info['name']} (Supply: {info['total_supply']})")
```

### View All Pools
```python
for pool_key, pool_info in system.pools.pools.items():
    print(f"Pool: {pool_key}")
    print(f"  Reserves: {pool_info['reserve0']} / {pool_info['reserve1']}")
```

### Get Account Info
```python
info = system.get_account_info(address)
print(f"Address: {info['address']}")
print(f"Balances: {info['balances']}")
```

## Configuration

### Database Location
Edit `main.py`:
```python
DATA_DIR = Path(__file__).parent / 'data'
```

### Logging Level
```python
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detail
    ...
)
```

### Default Tokens
Modify in `_initialize_defaults()`:
```python
self.tokens.create_token('NEW', 'New Token', Decimal('1000000'), 18)
```

## Troubleshooting

### Database Locked Error
- Ensure only one instance is running
- Check file permissions on `data/main.db`

### Import Error
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Permission Denied
```bash
chmod +x main.py
chmod -R 755 data/
```

## Next Steps

1. **Explore the System**
   - Try different token transfers
   - Create multiple accounts
   - Test swap calculations

2. **Read Documentation**
   - [System Overview](./SYSTEM_OVERVIEW.md)
   - [API Reference](./API_REFERENCE.md)

3. **Extend Functionality**
   - Add new tokens
   - Create custom pools
   - Implement additional features

## Support

For issues or questions:
- Check logs in `logs/system.log`
- Review error messages
- Consult documentation

## Security Notes

⚠️ **Development Only**
- This is a demonstration system
- Do not use for real funds
- Not audited for production

## Performance Tips

- Use decimal precision appropriately
- Batch operations when possible
- Monitor log file size
- Regular database maintenance