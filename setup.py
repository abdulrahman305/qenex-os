#!/usr/bin/env python3
"""
QENEX OS Setup Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="qenex-os",
    version="1.0.0",
    author="QENEX Team",
    author_email="support@qenex.ai",
    description="QENEX OS - Unified AI Operating System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdulrahman305/qenex-os",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "aiohttp>=3.8.0",
        "cryptography>=41.0.0",
        "numpy>=1.24.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "full": [
            "web3>=6.0.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "qenex-os=qenex_os.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)