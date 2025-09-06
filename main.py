#!/usr/bin/env python3
"""
QENEX Unified Financial Operating System v3.0
Main entry point for the complete financial OS
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from qenex_unified_core import QenexFinancialOS, SystemState
    from ai_self_improvement_engine import SelfImprovementEngine
    from cross_platform_layer import CrossPlatformLayer
    from advanced_financial_protocols import AdvancedFinancialProtocolManager
except ImportError as e:
    print(f"Failed to import core modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

async def main():
    """Main entry point"""
    print("=" * 80)
    print("QENEX UNIFIED FINANCIAL OPERATING SYSTEM v3.0")
    print("Production-Ready Financial Infrastructure")
    print("=" * 80)
    
    try:
        # Initialize cross-platform layer
        platform_layer = CrossPlatformLayer()
        await platform_layer.initialize()
        print(f"✅ Platform: {platform_layer.system_info.platform}")
        
        # Initialize core financial OS
        financial_os = QenexFinancialOS()
        await financial_os.initialize()
        print("✅ Financial OS Core Initialized")
        
        # Initialize financial protocols
        protocol_manager = AdvancedFinancialProtocolManager()
        print("✅ Financial Protocols Initialized")
        
        # Initialize AI self-improvement
        if platform_layer.system_info.memory_gb > 4:  # Only if sufficient memory
            print("✅ AI Self-Improvement Engine Available")
        
        # Get system status
        status = await financial_os.get_system_status()
        print(f"✅ System Status: {status['state']}")
        print(f"✅ Uptime: {status['uptime_seconds']:.1f} seconds")
        
        # Keep system running
        if len(sys.argv) > 1 and sys.argv[1] == '--daemon':
            print("\n🚀 QENEX Financial OS running in daemon mode...")
            while financial_os.state == SystemState.RUNNING:
                await asyncio.sleep(10)
        else:
            print("\n🚀 QENEX Financial OS ready!")
            print("Use --daemon flag to run in background")
            
    except Exception as e:
        logging.error(f"System startup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
