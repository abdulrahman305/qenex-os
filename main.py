#!/usr/bin/env python3
"""
QENEX Financial Operating System - Main Entry Point
Unified financial infrastructure platform
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

from aiohttp import web
from prometheus_client import start_http_server

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core components
try:
    from core.financial_kernel import FinancialKernel
    from core.payment_protocols import PaymentGateway
    from core.ai_engine import FraudDetector, RiskAssessmentEngine, SelfImprovementEngine
except ImportError:
    # Fallback imports for existing modules
    from qenex_unified_core import QenexFinancialOS as FinancialKernel
    PaymentGateway = None
    FraudDetector = None
    RiskAssessmentEngine = None
    SelfImprovementEngine = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QenexFinancialOS:
    """Main QENEX Financial Operating System"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize QENEX Financial OS"""
        self.config = self._load_config(config_path)
        self.running = False
        
        # Core components
        self.kernel = None
        self.payment_gateway = None
        self.fraud_detector = None
        self.risk_engine = None
        self.self_improvement = None
        
        # Web API
        self.app = web.Application()
        self.runner = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'user': 'qenex',
                'password': 'secure_password',
                'database': 'qenex_financial',
                'redis_host': 'localhost',
                'redis_port': 6379
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8080
            },
            'metrics': {
                'port': 9090
            },
            'workers': {
                'transaction_processors': 4,
                'payment_handlers': 2,
                'ai_workers': 2
            },
            'features': {
                'fraud_detection': True,
                'risk_assessment': True,
                'self_improvement': True,
                'real_time_processing': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def initialize(self):
        """Initialize all system components"""
        logger.info("Initializing QENEX Financial OS...")
        
        try:
            # Initialize financial kernel
            if FinancialKernel:
                self.kernel = FinancialKernel(self.config)
                await self.kernel.initialize()
                logger.info("✓ Financial kernel initialized")
            
            # Initialize payment gateway
            if PaymentGateway:
                self.payment_gateway = PaymentGateway()
                await self.payment_gateway.initialize()
                logger.info("✓ Payment gateway initialized")
            
            # Initialize AI components
            if FraudDetector and self.config['features']['fraud_detection']:
                self.fraud_detector = FraudDetector()
                logger.info("✓ Fraud detection engine initialized")
            
            if RiskAssessmentEngine and self.config['features']['risk_assessment']:
                self.risk_engine = RiskAssessmentEngine()
                logger.info("✓ Risk assessment engine initialized")
            
            if SelfImprovementEngine and self.config['features']['self_improvement']:
                self.self_improvement = SelfImprovementEngine()
                logger.info("✓ Self-improvement engine initialized")
            
            # Setup API routes
            self._setup_api_routes()
            
            # Start metrics server
            start_http_server(self.config['metrics']['port'])
            logger.info(f"✓ Metrics server started on port {self.config['metrics']['port']}")
            
            self.running = True
            logger.info("QENEX Financial OS initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    def _setup_api_routes(self):
        """Setup REST API routes"""
        
        # Health check
        self.app.router.add_get('/health', self.health_check)
        
        # System status
        self.app.router.add_get('/api/v1/system/status', self.system_status)
    
    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint"""
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0'
        })
    
    async def system_status(self, request: web.Request) -> web.Response:
        """Get system status"""
        status = {
            'running': self.running,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'components': {
                'kernel': 'active' if self.kernel else 'inactive',
                'payment_gateway': 'active' if self.payment_gateway else 'inactive',
                'fraud_detector': 'active' if self.fraud_detector else 'inactive',
                'risk_engine': 'active' if self.risk_engine else 'inactive',
                'self_improvement': 'active' if self.self_improvement else 'inactive'
            },
            'config': {
                'workers': self.config['workers'],
                'features': self.config['features']
            }
        }
        
        return web.json_response(status)
    
    async def start(self):
        """Start the financial OS"""
        await self.initialize()
        
        # Setup web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        site = web.TCPSite(
            self.runner,
            self.config['api']['host'],
            self.config['api']['port']
        )
        
        await site.start()
        logger.info(f"API server started on {self.config['api']['host']}:{self.config['api']['port']}")
        
        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the system gracefully"""
        logger.info("Shutting down QENEX Financial OS...")
        self.running = False
        
        # Cleanup web server
        if self.runner:
            await self.runner.cleanup()
        
        # Shutdown kernel
        if self.kernel and hasattr(self.kernel, 'shutdown'):
            await self.kernel.shutdown()
        
        logger.info("QENEX Financial OS shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='QENEX Financial Operating System')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon'
    )
    
    args = parser.parse_args()
    
    # ASCII Art Banner
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ██████╗ ███████╗███╗   ██╗███████╗██╗  ██╗            ║
    ║  ██╔═══██╗██╔════╝████╗  ██║██╔════╝╚██╗██╔╝            ║
    ║  ██║   ██║█████╗  ██╔██╗ ██║█████╗   ╚███╔╝             ║
    ║  ██║▄▄ ██║██╔══╝  ██║╚██╗██║██╔══╝   ██╔██╗             ║
    ║  ╚██████╔╝███████╗██║ ╚████║███████╗██╔╝ ██╗            ║
    ║   ╚══▀▀═╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝            ║
    ║                                                           ║
    ║          Financial Operating System v1.0.0               ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    
    print(banner)
    
    # Create and run the system
    qenex = QenexFinancialOS(args.config)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        qenex.running = False
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the system
    try:
        asyncio.run(qenex.start())
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
