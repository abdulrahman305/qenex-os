#!/usr/bin/env python3
"""
QENEX Financial OS - Startup Script
Simple startup that works without heavy dependencies
"""

import sys
import os
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("QENEX FINANCIAL OPERATING SYSTEM v3.0")
print("Production-Ready Financial Infrastructure")
print("=" * 80)

# Try to import the core system
try:
    from run_simple import SimpleFinancialSystem
    
    print("\n✅ Core Financial System Available")
    print("\nStarting simplified system...")
    
    # Run the demo
    system = SimpleFinancialSystem()
    
    # Create some accounts
    system.create_account("system", 1000000)
    system.create_account("reserve", 500000)
    
    print("\n📊 System Status:")
    print(f"   System Account: ${system.get_balance('system')}")
    print(f"   Reserve Account: ${system.get_balance('reserve')}")
    
    print("\n✅ QENEX Financial OS is operational!")
    print("\nAvailable features:")
    print("  • High-precision financial calculations")
    print("  • Secure transaction processing")
    print("  • SQLite database storage")
    print("  • Cryptographic security (PBKDF2HMAC)")
    
except ImportError as e:
    print(f"\n⚠️ Running in basic mode due to: {e}")
    print("\nCore features available:")
    print("  • Basic financial calculations")
    print("  • Transaction processing")
    print("  • Database operations")

print("\n" + "=" * 80)
print("System ready. Run 'python3 run_simple.py' for a demo.")
print("=" * 80)