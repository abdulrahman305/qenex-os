from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qenex-financial-os",
    version="3.0.0",
    author="QENEX Team",
    author_email="contact@qenex.ai",
    description="QENEX Unified Financial Operating System - Production-Ready Financial Infrastructure",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdulrahman305/qenex-os",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qenex=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
