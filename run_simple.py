#!/usr/bin/env python3
"""
QENEX Financial OS - Simple Demo
Demonstrates core functionality without heavy dependencies
"""

import os
import json
import sqlite3
from decimal import Decimal, getcontext
from datetime import datetime
import hashlib

# Set high precision for financial calculations
getcontext().prec = 28

print("=" * 60)
print("QENEX FINANCIAL OS - SIMPLE DEMO")
print("=" * 60)

class SimpleFinancialSystem:
    def __init__(self):
        self.db = sqlite3.connect(':memory:')
        self.setup_database()
        self.accounts = {}
        
    def setup_database(self):
        cursor = self.db.cursor()
        cursor.execute('''
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_account TEXT,
                to_account TEXT,
                amount REAL,
                timestamp TEXT,
                status TEXT
            )
        ''')
        self.db.commit()
        
    def create_account(self, account_id, initial_balance=0):
        self.accounts[account_id] = Decimal(str(initial_balance))
        print(f"✅ Created account {account_id} with balance: {initial_balance}")
        
    def transfer(self, from_acc, to_acc, amount):
        amount = Decimal(str(amount))
        if from_acc not in self.accounts:
            print(f"❌ Account {from_acc} not found")
            return False
            
        if self.accounts[from_acc] < amount:
            print(f"❌ Insufficient balance in {from_acc}")
            return False
            
        # Execute transfer
        self.accounts[from_acc] -= amount
        self.accounts[to_acc] = self.accounts.get(to_acc, Decimal('0')) + amount
        
        # Record transaction
        cursor = self.db.cursor()
        cursor.execute(
            "INSERT INTO transactions (from_account, to_account, amount, timestamp, status) VALUES (?, ?, ?, ?, ?)",
            (from_acc, to_acc, float(amount), datetime.now().isoformat(), "completed")
        )
        self.db.commit()
        
        print(f"✅ Transferred {amount} from {from_acc} to {to_acc}")
        return True
        
    def get_balance(self, account_id):
        return self.accounts.get(account_id, Decimal('0'))
        
    def get_transaction_history(self):
        cursor = self.db.cursor()
        cursor.execute("SELECT * FROM transactions ORDER BY timestamp DESC LIMIT 10")
        return cursor.fetchall()

# Demo the system
print("\n🚀 Starting QENEX Financial System Demo...\n")

system = SimpleFinancialSystem()

# Create accounts
system.create_account("alice", 10000)
system.create_account("bob", 5000)
system.create_account("charlie", 2500)

print("\n📊 Initial Balances:")
for acc in ["alice", "bob", "charlie"]:
    print(f"   {acc}: {system.get_balance(acc)}")

# Perform transactions
print("\n💸 Executing Transactions:")
system.transfer("alice", "bob", 1500)
system.transfer("bob", "charlie", 750)
system.transfer("charlie", "alice", 250)

print("\n📊 Final Balances:")
for acc in ["alice", "bob", "charlie"]:
    print(f"   {acc}: {system.get_balance(acc)}")

print("\n📜 Transaction History:")
for tx in system.get_transaction_history():
    print(f"   TX #{tx[0]}: {tx[1]} → {tx[2]}: ${tx[3]} [{tx[5]}]")

print("\n" + "=" * 60)
print("✅ QENEX Financial OS Demo Complete!")
print("=" * 60)