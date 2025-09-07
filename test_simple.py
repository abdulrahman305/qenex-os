#!/usr/bin/env python3
"""
Simple test to verify the QENEX system works
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_simple import SimpleFinancialSystem

class TestSimpleFinancialSystem(unittest.TestCase):
    """Test the simple financial system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = SimpleFinancialSystem()
    
    def test_create_account(self):
        """Test account creation"""
        self.system.create_account("test_user", 1000)
        balance = self.system.get_balance("test_user")
        self.assertEqual(balance, 1000)
    
    def test_transfer(self):
        """Test money transfer"""
        self.system.create_account("alice", 1000)
        self.system.create_account("bob", 500)
        
        # Transfer 200 from alice to bob
        result = self.system.transfer("alice", "bob", 200)
        self.assertTrue(result)
        
        # Check balances
        self.assertEqual(self.system.get_balance("alice"), 800)
        self.assertEqual(self.system.get_balance("bob"), 700)
    
    def test_insufficient_funds(self):
        """Test transfer with insufficient funds"""
        self.system.create_account("alice", 100)
        self.system.create_account("bob", 0)
        
        # Try to transfer more than available
        result = self.system.transfer("alice", "bob", 200)
        self.assertFalse(result)
        
        # Balances should remain unchanged
        self.assertEqual(self.system.get_balance("alice"), 100)
        self.assertEqual(self.system.get_balance("bob"), 0)
    
    def test_transaction_history(self):
        """Test transaction history recording"""
        self.system.create_account("alice", 1000)
        self.system.create_account("bob", 500)
        
        # Make some transactions
        self.system.transfer("alice", "bob", 100)
        self.system.transfer("bob", "alice", 50)
        
        # Check history
        history = self.system.get_transaction_history()
        self.assertEqual(len(history), 2)
        
        # Most recent transaction should be first
        self.assertEqual(history[0][1], "bob")  # from_account
        self.assertEqual(history[0][2], "alice")  # to_account
        self.assertEqual(history[0][3], 50.0)  # amount

if __name__ == '__main__':
    unittest.main()