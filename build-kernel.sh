#!/bin/bash
# QENEX Banking OS Kernel Build Script
# Builds bootable ISO for production banking deployment

set -e  # Exit on any error

echo "QENEX Banking OS - Kernel Build System"
echo "======================================"

# Check for required tools
echo "Checking build dependencies..."
for tool in as clang ld objcopy cargo grub-mkrescue qemu-system-x86_64; do
    if ! command -v $tool &> /dev/null; then
        echo "ERROR: $tool is required but not installed"
        exit 1
    fi
done

# Check for banking-specific tools
echo "Checking banking security tools..."
if ! command -v tpm2_startup &> /dev/null; then
    echo "WARNING: TPM tools not found - some security features disabled"
fi

# Create build directories
echo "Creating build directories..."
mkdir -p build/{kernel,iso/boot/grub}

# Build assembly components
echo "Compiling boot assembly..."
as --64 -g src/kernel/boot.s -o build/kernel/boot.o

# Compile Rust kernel with banking features
echo "Building Rust kernel with banking security..."
export RUSTFLAGS="-C target-cpu=x86-64 -C target-feature=-mmx,-sse,+soft-float -C relocation-model=static -C code-model=kernel -C panic=abort --cfg banking_kernel"
cargo build --target x86_64-unknown-none --features kernel --no-default-features --release

# Link kernel
echo "Linking banking kernel..."
ld -n -T kernel.ld --gc-sections -o build/qenex-banking-kernel.elf \
   build/kernel/boot.o \
   target/x86_64-unknown-none/release/libqenex_os.a

# Create kernel binary
echo "Creating kernel binary..."
objcopy -O binary build/qenex-banking-kernel.elf build/qenex-banking-kernel.bin

# Copy kernel to ISO directory
cp build/qenex-banking-kernel.bin build/iso/boot/

# Create GRUB configuration
cat > build/iso/boot/grub/grub.cfg << 'EOF'
set timeout=3
set default=0

menuentry "QENEX Banking OS - Production" {
    multiboot2 /boot/qenex-banking-kernel.bin
    boot
}

menuentry "QENEX Banking OS - Safe Mode" {
    multiboot2 /boot/qenex-banking-kernel.bin safe_mode=1
    boot
}

menuentry "QENEX Banking OS - Recovery" {
    multiboot2 /boot/qenex-banking-kernel.bin recovery_mode=1
    boot
}
EOF

# Build bootable ISO
echo "Creating bootable ISO..."
grub-mkrescue -o qenex-banking-os.iso build/iso/

# Verify ISO
if [ -f qenex-banking-os.iso ]; then
    echo "SUCCESS: Bootable banking OS ISO created: qenex-banking-os.iso"
    echo "Size: $(du -h qenex-banking-os.iso | cut -f1)"
    
    # Security verification
    echo "Performing security verification..."
    objdump -h build/qenex-banking-kernel.elf | grep -E "(LOAD|EXEC)"
    
    echo ""
    echo "Banking OS Build Complete!"
    echo "========================="
    echo "ISO: qenex-banking-os.iso"
    echo "Kernel: build/qenex-banking-kernel.elf"
    echo ""
    echo "Test with: qemu-system-x86_64 -cdrom qenex-banking-os.iso -m 1G"
    echo "Deploy to production banking hardware with proper security protocols"
else
    echo "ERROR: Failed to create bootable ISO"
    exit 1
fi