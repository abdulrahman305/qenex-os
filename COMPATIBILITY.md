# QENEX OS Compatibility Guide

## System Requirements

### Operating Systems
- ✅ **Linux**: Ubuntu 20.04+, Debian 11+, CentOS 8+, Fedora 34+, Arch Linux
- ✅ **macOS**: 10.15 (Catalina) or later
- ✅ **Windows**: Windows 10 (version 1909+), Windows 11, Windows Server 2019+

### Python Versions
- ✅ Python 3.8
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12 (experimental)

### Hardware Requirements
- **Minimum**:
  - CPU: 2 cores, 2.0 GHz
  - RAM: 4 GB
  - Storage: 1 GB available space
  - Network: Internet connection for blockchain features

- **Recommended**:
  - CPU: 4+ cores, 3.0 GHz
  - RAM: 8 GB or more
  - Storage: 10 GB available space
  - Network: Broadband connection

## Dependency Compatibility

### Core Dependencies
| Package | Min Version | Max Version | Notes |
|---------|------------|-------------|-------|
| aiohttp | 3.8.0 | 3.9.x | Async HTTP client/server |
| cryptography | 41.0.0 | 42.x.x | Encryption and security |
| numpy | 1.24.0 | 1.26.x | Numerical computing |
| psutil | 5.9.0 | 5.9.x | System monitoring |
| pyyaml | 6.0 | 6.0.x | YAML parsing |
| requests | 2.31.0 | 2.32.x | HTTP library |

### Optional Dependencies
| Package | Min Version | Notes |
|---------|------------|-------|
| web3 | 6.0.0 | Blockchain integration |
| pandas | 2.0.0 | Data analysis |
| scikit-learn | 1.3.0 | Machine learning |

## Installation Methods

### Linux/macOS
```bash
# Method 1: Using install script
sudo ./install.sh

# Method 2: Using pip
pip install -e .

# Method 3: Using setup.py
python setup.py install
```

### Windows
```batch
# Method 1: Using install batch file
install.bat

# Method 2: Using pip
pip install -e .

# Method 3: Manual installation
python setup.py install
```

## Cross-Platform Considerations

### File Paths
- Uses `pathlib.Path` for cross-platform path handling
- Automatically detects OS and adjusts paths accordingly
- Supports both forward and backward slashes

### Process Management
- Uses `psutil` for cross-platform process handling
- Gracefully handles platform-specific features
- Falls back to basic functionality on unsupported platforms

### Network Operations
- Uses `aiohttp` for async networking
- Compatible with all major platforms
- Handles platform-specific socket behaviors

### Security Features
- Cryptography library works on all platforms
- Platform-agnostic encryption/decryption
- Cross-platform password hashing

## Known Compatibility Issues and Solutions

### Issue 1: Python 3.7 and below
**Problem**: Type hints use newer syntax not available in Python 3.7
**Solution**: Upgrade to Python 3.8 or higher

### Issue 2: Windows Path Length
**Problem**: Windows has a 260-character path limit by default
**Solution**: Enable long path support in Windows or install to a shorter path

### Issue 3: macOS Security Restrictions
**Problem**: macOS may block unsigned executables
**Solution**: Use `xattr -d com.apple.quarantine qenex-os` after installation

### Issue 4: Linux Permissions
**Problem**: Installation requires root permissions
**Solution**: Use `sudo` for installation or install in user directory

## Testing Matrix

| OS | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|----|------------|------------|-------------|-------------|-------------|
| Ubuntu 20.04 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Ubuntu 22.04 | ✅ | ✅ | ✅ | ✅ | ✅ |
| macOS 12 | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| macOS 13 | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Windows 10 | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Windows 11 | ✅ | ✅ | ✅ | ✅ | ⚠️ |

Legend: ✅ Fully Supported | ⚠️ Experimental | ❌ Not Supported

## Browser Compatibility (Web Interface)

### Supported Browsers
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## API Compatibility

### REST API
- Supports JSON content type
- Compatible with HTTP/1.1 and HTTP/2
- CORS enabled for cross-origin requests

### WebSocket
- Supports WSS (WebSocket Secure)
- Compatible with Socket.IO clients
- Auto-reconnection on connection loss

## Backward Compatibility

### Version Policy
- Semantic versioning (MAJOR.MINOR.PATCH)
- Breaking changes only in major versions
- Deprecation warnings before removal
- Migration guides for major updates

### Configuration Files
- Supports legacy configuration formats
- Automatic migration from old formats
- Backward-compatible default values

## Future Compatibility

### Planned Support
- Python 3.13 (when released)
- ARM64 architecture optimization
- Docker and Kubernetes deployment
- WebAssembly compilation

### Deprecation Schedule
- Python 3.8 support until 2024-10
- Legacy API endpoints until v2.0.0
- Old configuration format until v1.5.0

## Getting Help

### Compatibility Issues
1. Check this guide first
2. Search [existing issues](https://github.com/abdulrahman305/qenex-os/issues)
3. Create a new issue with:
   - OS version
   - Python version
   - Error messages
   - Steps to reproduce

### Community Support
- Discord: [Join our server](https://discord.gg/qenex)
- GitHub Discussions: [Ask questions](https://github.com/abdulrahman305/qenex-os/discussions)
- Email: ceo@qenex.ai

## Contributing

Help us improve compatibility:
1. Test on your platform
2. Report issues
3. Submit pull requests
4. Update documentation

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.