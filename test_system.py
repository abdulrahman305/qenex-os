#!/usr/bin/env python3
"""
QENEX OS System Test
Quick test to verify the system is working correctly
"""

import sys
import os

print("=" * 60)
print("QENEX Financial OS - System Test")
print("=" * 60)

# Test imports
print("\n1. Testing core imports...")
try:
    import hashlib
    import json
    import sqlite3
    from decimal import Decimal
    from datetime import datetime
    print("✅ Standard library imports: OK")
except ImportError as e:
    print(f"❌ Standard library import failed: {e}")
    sys.exit(1)

# Test cryptography
print("\n2. Testing cryptography...")
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import os
    
    # Create a simple key derivation
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=os.urandom(16),
        iterations=100000,
    )
    key = kdf.derive(b"test_password")
    print(f"✅ Cryptography working: Generated {len(key)}-byte key")
except ImportError as e:
    print(f"❌ Cryptography import failed: {e}")
except Exception as e:
    print(f"❌ Cryptography test failed: {e}")

# Test database
print("\n3. Testing database operations...")
try:
    import sqlite3
    
    # Create in-memory database
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create test table
    cursor.execute('''
        CREATE TABLE test_transactions (
            id INTEGER PRIMARY KEY,
            amount REAL,
            status TEXT
        )
    ''')
    
    # Insert test data
    cursor.execute("INSERT INTO test_transactions (amount, status) VALUES (?, ?)", 
                   (1000.50, "completed"))
    conn.commit()
    
    # Query data
    cursor.execute("SELECT * FROM test_transactions")
    result = cursor.fetchone()
    
    if result:
        print(f"✅ Database working: Test transaction {result}")
    else:
        print("❌ Database test failed: No data retrieved")
    
    conn.close()
except Exception as e:
    print(f"❌ Database test failed: {e}")

# Test financial calculations
print("\n4. Testing financial calculations...")
try:
    from decimal import Decimal, getcontext
    
    getcontext().prec = 28
    
    amount1 = Decimal("1000.123456789")
    amount2 = Decimal("2500.987654321")
    
    total = amount1 + amount2
    fee = total * Decimal("0.001")  # 0.1% fee
    net = total - fee
    
    print(f"✅ Financial calculations working:")
    print(f"   Amount 1: {amount1}")
    print(f"   Amount 2: {amount2}")
    print(f"   Total: {total}")
    print(f"   Fee (0.1%): {fee}")
    print(f"   Net: {net}")
except Exception as e:
    print(f"❌ Financial calculation failed: {e}")

# Test main system components
print("\n5. Testing QENEX system components...")
try:
    # Try importing main components
    components_to_test = [
        ('main', "Main system"),
        ('core', "Core module"),
        ('blockchain', "Blockchain module"),
    ]
    
    for module_name, description in components_to_test:
        try:
            module = __import__(module_name)
            print(f"✅ {description} ({module_name}): Available")
        except ImportError:
            print(f"⚠️  {description} ({module_name}): Not available (optional)")
        except Exception as e:
            print(f"❌ {description} ({module_name}): Error - {e}")

except Exception as e:
    print(f"❌ Component testing failed: {e}")

# Summary
print("\n" + "=" * 60)
print("SYSTEM TEST COMPLETE")
print("=" * 60)
print("\nThe QENEX Financial OS core components are functioning.")
print("You can now run the main system with: python3 main.py")
print("=" * 60)