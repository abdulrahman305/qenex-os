#!/usr/bin/env python3
"""
Mock dependencies for testing when real packages are not available
"""

import sys
from unittest.mock import MagicMock

# Create mock modules for missing dependencies
sys.modules['asyncpg'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['web3'] = MagicMock()

# Mock common classes/functions
class MockWeb3:
    def __init__(self, *args, **kwargs):
        pass
    
    class HTTPProvider:
        def __init__(self, *args, **kwargs):
            pass

sys.modules['web3'].Web3 = MockWeb3
sys.modules['web3'].HTTPProvider = MockWeb3.HTTPProvider

# Mock TensorFlow components
sys.modules['tensorflow'].keras = MagicMock()
sys.modules['tensorflow'].keras.Sequential = MagicMock
sys.modules['tensorflow'].keras.layers = MagicMock()

print("Mock dependencies loaded successfully")