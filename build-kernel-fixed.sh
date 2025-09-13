#!/bin/bash

# QENEX Banking OS - Fixed Build System
echo "QENEX Banking OS - Fixed Build System"
echo "===================================="

# Set up environment
export RUST_BACKTRACE=1

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "ERROR: Cargo.toml not found. Please run from project root."
    exit 1
fi

# Function to check command availability
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 not found"
        return 1
    fi
    echo "✓ $1 found"
    return 0
}

echo "Checking dependencies..."
check_command rustc || exit 1
check_command cargo || exit 1

# Add required targets
echo "Adding required Rust targets..."
rustup target add x86_64-unknown-linux-gnu 2>/dev/null || true
rustup target add x86_64-unknown-none 2>/dev/null || true

echo "Creating build directories..."
mkdir -p target/kernel
mkdir -p target/userspace
mkdir -p logs

echo ""
echo "=== Build Options ==="
echo "1. Userspace only (recommended)"
echo "2. Kernel only (bare-metal)"
echo "3. Both userspace and kernel"
echo "4. Run Python version instead"
echo ""

read -p "Choose option [1-4]: " choice

case $choice in
    1|"")
        echo "Building userspace components..."
        echo ""

        # Build userspace library
        echo "Building core userspace library..."
        cargo build --lib --features userspace,networking,database,crypto --target x86_64-unknown-linux-gnu 2>&1 | tee logs/userspace-build.log

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Userspace library built successfully"

            # Build API server
            echo "Building API server..."
            cargo build --bin qenex-api-server --features userspace,networking,database --target x86_64-unknown-linux-gnu 2>&1 | tee -a logs/userspace-build.log

            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "✓ API server built successfully"
                echo ""
                echo "To run the system:"
                echo "  ./target/x86_64-unknown-linux-gnu/debug/qenex-api-server"
            else
                echo "✗ API server build failed (check logs/userspace-build.log)"
            fi
        else
            echo "✗ Userspace build failed (check logs/userspace-build.log)"
            echo ""
            echo "Trying Python fallback..."
            python3 main.py || python3 run_simple.py
        fi
        ;;

    2)
        echo "Building kernel components..."
        echo ""

        cargo build --bin qenex-kernel --features kernel --target x86_64-unknown-none --profile kernel 2>&1 | tee logs/kernel-build.log

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Kernel built successfully"
            echo "Kernel binary: target/x86_64-unknown-none/kernel/qenex-kernel"
        else
            echo "✗ Kernel build failed (check logs/kernel-build.log)"
        fi
        ;;

    3)
        echo "Building both userspace and kernel..."
        echo ""

        # Userspace first
        cargo build --lib --features userspace,networking,database --target x86_64-unknown-linux-gnu 2>&1 | tee logs/userspace-build.log
        userspace_result=${PIPESTATUS[0]}

        # Kernel second
        cargo build --bin qenex-kernel --features kernel --target x86_64-unknown-none --profile kernel 2>&1 | tee logs/kernel-build.log
        kernel_result=${PIPESTATUS[0]}

        echo ""
        echo "Build Results:"
        if [ $userspace_result -eq 0 ]; then
            echo "✓ Userspace: SUCCESS"
        else
            echo "✗ Userspace: FAILED"
        fi

        if [ $kernel_result -eq 0 ]; then
            echo "✓ Kernel: SUCCESS"
        else
            echo "✗ Kernel: FAILED"
        fi
        ;;

    4)
        echo "Running Python version..."
        echo ""

        if [ -f "requirements.txt" ]; then
            echo "Installing Python dependencies..."
            pip3 install -r requirements.txt 2>/dev/null || echo "Warning: pip install failed"
        fi

        if [ -f "main.py" ]; then
            echo "Starting main.py..."
            python3 main.py
        elif [ -f "run_simple.py" ]; then
            echo "Starting run_simple.py..."
            python3 run_simple.py
        else
            echo "No Python entry point found"
        fi
        ;;

    *)
        echo "Invalid choice. Defaulting to userspace build."
        cargo build --lib --features userspace,networking,database --target x86_64-unknown-linux-gnu
        ;;
esac

echo ""
echo "Build process completed."
echo "Logs available in logs/ directory"