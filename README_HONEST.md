# QENEX Framework - Python Application Suite (Not an OS)

## ⚠️ IMPORTANT DISCLAIMER
**This is NOT an operating system**. It's a Python application that simulates OS-like concepts for educational and development purposes.

## What This Actually Is

QENEX is a Python framework that demonstrates:
- Basic neural network implementation
- Simple encryption/decryption utilities  
- Process monitoring (read-only)
- File management helpers
- Mock blockchain interaction
- Simulated DeFi operations

## What This Is NOT

- ❌ NOT a real operating system
- ❌ NOT capable of managing hardware
- ❌ NOT providing real security
- ❌ NOT performing real blockchain transactions
- ❌ NOT actually managing memory or processes
- ❌ NOT a replacement for your OS

## Actual Capabilities

### ✅ Working Features:
1. **Basic Neural Network** - Simple feedforward network (requires training data)
2. **File Operations** - Read/write files using Python
3. **Process Listing** - View running processes (read-only via psutil)
4. **Basic Encryption** - Fernet symmetric encryption
5. **CLI Interface** - Command-line menu system

### ❌ Simulated/Mock Features:
1. **Kernel** - Just Python classes, no kernel access
2. **Memory Management** - Dictionary-based simulation
3. **Network Stack** - Returns dummy data
4. **Blockchain** - Points to demo endpoints (non-functional)
5. **DeFi Operations** - Number manipulation only
6. **P2P Networking** - Completely simulated
7. **Threat Detection** - Hardcoded string matching

## Installation

```bash
# Install as a Python package
pip install -r requirements.txt
python qenex-os  # Runs the CLI
```

## System Requirements

- Python 3.8+
- 100MB disk space
- No special privileges required (it's just a Python app)

## Use Cases

✅ **Good for:**
- Learning about OS concepts
- Python programming examples
- Understanding neural networks basics
- CLI application development

❌ **NOT for:**
- Production systems
- Real security
- Actual cryptocurrency operations
- System administration
- Process management

## Example Usage

```python
# Use the neural network component
from core.ai.ai_engine import NeuralNetwork
nn = NeuralNetwork()
# Note: You need to provide your own training data

# Use basic encryption
from core.security.security_manager import SecurityManager
sm = SecurityManager()
encrypted = sm.encrypt_data(b"hello")  # Basic encryption only
```

## Known Limitations

1. **No Real OS Functions** - Cannot control hardware, manage real processes, or allocate memory
2. **No Blockchain Connection** - All crypto operations are simulated
3. **No Network Stack** - Cannot send/receive real network packets
4. **No Security** - Basic encryption only, no real threat protection
5. **No AI Learning** - Neural network exists but needs external training

## Development Status

This is a **demonstration project** showing Python implementations of OS-like concepts. It should not be used for any production purpose or relied upon for actual system operations.

## Contributing

Feel free to contribute, but please:
- Don't claim it's a real OS
- Don't add more fake functionality
- Focus on educational value
- Be honest about limitations

## License

MIT - Use at your own risk. This software comes with NO WARRANTIES and should not be used for any critical operations.

## Support

This is an educational project. For actual OS needs, use:
- Linux (Ubuntu, Debian, etc.)
- Windows
- macOS

For actual DeFi operations, use established platforms with real smart contracts.

---

**Remember**: This is a Python simulation, NOT a real operating system. Never rely on it for actual system operations, security, or financial transactions.