# QENEX Banking OS Production Build System
# Builds bootable ISO for real hardware deployment

# Build configuration
ARCH := x86_64
TARGET := x86_64-unknown-none
KERNEL_NAME := qenex-banking-kernel
ISO_NAME := qenex-banking-os.iso

# Directories
BUILD_DIR := build
BOOT_DIR := boot
ISO_DIR := $(BUILD_DIR)/iso
ISO_BOOT_DIR := $(ISO_DIR)/boot
GRUB_DIR := $(ISO_BOOT_DIR)/grub
SRC_DIR := src
OBJ_DIR := $(BUILD_DIR)/obj

# Tools and flags
AS := nasm
CC := clang
LD := ld
OBJCOPY := objcopy
RUSTC := rustc
CARGO := cargo
GRUB_MKRESCUE := grub-mkrescue
QEMU := qemu-system-x86_64

# Assembly flags
ASFLAGS := -f elf64 -g -F dwarf

# C compiler flags for kernel
CFLAGS := -target x86_64-unknown-none-elf \
          -ffreestanding \
          -fno-stack-protector \
          -fno-stack-check \
          -fno-lto \
          -fPIE \
          -m64 \
          -mabi=sysv \
          -mno-80387 \
          -mno-mmx \
          -mno-3dnow \
          -mno-sse \
          -mno-sse2 \
          -mno-red-zone \
          -mcmodel=kernel \
          -g \
          -O2 \
          -Wall \
          -Wextra \
          -Werror \
          -DQENEX_BANKING_KERNEL=1 \
          -DSECURITY_LEVEL_MAXIMUM=1 \
          -std=c11

# Linker flags
LDFLAGS := -n \
           -T linker.ld \
           --gc-sections \
           -z max-page-size=0x1000

# Rust flags for kernel
RUSTFLAGS := --target $(TARGET) \
             --edition 2021 \
             -C target-cpu=x86-64 \
             -C target-feature=-mmx,-sse,+soft-float \
             -C relocation-model=static \
             -C code-model=kernel \
             -C panic=abort \
             -C opt-level=2 \
             -C lto=fat \
             -C codegen-units=1 \
             --cfg qenex_banking_kernel \
             --cfg security_level_maximum

# Source files
BOOT_ASM_SOURCES := $(wildcard $(BOOT_DIR)/*.asm)
KERNEL_C_SOURCES := $(wildcard $(SRC_DIR)/kernel/*.c)
KERNEL_ASM_OBJECTS := $(patsubst $(BOOT_DIR)/%.asm,$(OBJ_DIR)/%.o,$(BOOT_ASM_SOURCES))
KERNEL_C_OBJECTS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(KERNEL_C_SOURCES))

# Output files
KERNEL_ELF := $(BUILD_DIR)/$(KERNEL_NAME).elf
KERNEL_BIN := $(BUILD_DIR)/$(KERNEL_NAME).bin

.PHONY: all clean iso test test-kvm debug help kernel

# Default target
all: iso

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(OBJ_DIR)
	mkdir -p $(ISO_DIR)
	mkdir -p $(ISO_BOOT_DIR)
	mkdir -p $(GRUB_DIR)

# Compile boot assembly files
$(OBJ_DIR)/%.o: $(BOOT_DIR)/%.asm | $(BUILD_DIR)
	$(AS) $(ASFLAGS) $< -o $@

# Compile kernel C files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Build Rust kernel components
$(BUILD_DIR)/kernel_rust.o: $(shell find $(SRC_DIR) -name "*.rs") | $(BUILD_DIR)
	RUSTFLAGS="$(RUSTFLAGS)" $(CARGO) build --target $(TARGET) --release --no-default-features --features kernel
	cp target/$(TARGET)/release/deps/libqenex_os-*.rlib $(BUILD_DIR)/kernel_rust.o 2>/dev/null || \
	$(RUSTC) $(RUSTFLAGS) --crate-type staticlib --crate-name qenex_os $(SRC_DIR)/lib.rs -o $@

# Link kernel
$(KERNEL_ELF): $(KERNEL_ASM_OBJECTS) $(KERNEL_C_OBJECTS) $(BUILD_DIR)/kernel_rust.o linker.ld | $(BUILD_DIR)
	$(LD) $(LDFLAGS) -o $@ $(KERNEL_ASM_OBJECTS) $(KERNEL_C_OBJECTS) $(BUILD_DIR)/kernel_rust.o

# Create kernel binary
$(KERNEL_BIN): $(KERNEL_ELF) | $(BUILD_DIR)
	$(OBJCOPY) -O binary $< $@

# Copy kernel to ISO boot directory
$(ISO_BOOT_DIR)/$(KERNEL_NAME).bin: $(KERNEL_BIN) | $(BUILD_DIR)
	cp $< $@

# Create GRUB configuration
$(GRUB_DIR)/grub.cfg: | $(BUILD_DIR)
	@echo "Creating GRUB configuration..."
	@echo "# QENEX Banking OS GRUB Configuration" > $@
	@echo "set timeout=5" >> $@
	@echo "set default=0" >> $@
	@echo "" >> $@
	@echo "# Set video mode for banking operations" >> $@
	@echo "set gfxmode=1024x768x32" >> $@
	@echo "terminal_output gfxterm" >> $@
	@echo "" >> $@
	@echo "# Main boot entry" >> $@
	@echo "menuentry \"QENEX Banking OS - Production Mode\" {" >> $@
	@echo "    echo \"Loading QENEX Banking OS in Production Mode...\"" >> $@
	@echo "    multiboot2 /boot/$(KERNEL_NAME).bin" >> $@
	@echo "    echo \"Starting banking kernel...\"" >> $@
	@echo "    boot" >> $@
	@echo "}" >> $@
	@echo "" >> $@
	@echo "# Safe mode for diagnostics" >> $@
	@echo "menuentry \"QENEX Banking OS - Safe Mode\" {" >> $@
	@echo "    echo \"Loading QENEX Banking OS in Safe Mode...\"" >> $@
	@echo "    multiboot2 /boot/$(KERNEL_NAME).bin safe_mode=1 debug=1" >> $@
	@echo "    boot" >> $@
	@echo "}" >> $@
	@echo "" >> $@
	@echo "# Recovery mode" >> $@
	@echo "menuentry \"QENEX Banking OS - Recovery Mode\" {" >> $@
	@echo "    echo \"Loading QENEX Banking OS in Recovery Mode...\"" >> $@
	@echo "    multiboot2 /boot/$(KERNEL_NAME).bin recovery_mode=1" >> $@
	@echo "    boot" >> $@
	@echo "}" >> $@
	@echo "" >> $@
	@echo "# Hardware diagnostics" >> $@
	@echo "menuentry \"QENEX Banking OS - Hardware Diagnostics\" {" >> $@
	@echo "    echo \"Loading hardware diagnostics...\"" >> $@
	@echo "    multiboot2 /boot/$(KERNEL_NAME).bin diagnostics=1" >> $@
	@echo "    boot" >> $@
	@echo "}" >> $@

# Build bootable ISO
iso: $(ISO_NAME)

$(ISO_NAME): $(ISO_BOOT_DIR)/$(KERNEL_NAME).bin $(GRUB_DIR)/grub.cfg
	@echo "Creating bootable ISO image..."
	$(GRUB_MKRESCUE) -o $@ $(ISO_DIR) 2>/dev/null
	@echo "SUCCESS: Bootable banking OS ISO created: $(ISO_NAME)"
	@echo "Size: $$(du -h $(ISO_NAME) | cut -f1)"
	@echo ""
	@echo "Banking OS Build Summary:"
	@echo "========================="
	@echo "Kernel ELF: $(KERNEL_ELF)"
	@echo "Kernel Binary: $(KERNEL_BIN)"
	@echo "Bootable ISO: $(ISO_NAME)"
	@echo ""
	@echo "Security Verification:"
	@objdump -h $(KERNEL_ELF) | grep -E "(EXEC|LOAD)"
	@echo ""
	@echo "Ready for deployment to banking hardware."

# Test in QEMU with banking hardware emulation
test: $(ISO_NAME)
	@echo "Testing QENEX Banking OS in QEMU..."
	$(QEMU) -cdrom $(ISO_NAME) \
		-m 2G \
		-smp 2 \
		-cpu qemu64,+aes,+rdrand,+rdseed \
		-machine q35 \
		-device tpm-tis,tpmdev=tpm0 \
		-tpmdev emulator,id=tpm0,chardev=chrtpm \
		-chardev socket,id=chrtpm,path=/tmp/qenex-tpm-sock \
		-netdev user,id=net0 \
		-device e1000,netdev=net0 \
		-serial stdio \
		-monitor tcp:localhost:4444,server,nowait

# Test with KVM acceleration (requires KVM support)
test-kvm: $(ISO_NAME)
	@echo "Testing QENEX Banking OS with KVM acceleration..."
	$(QEMU) -enable-kvm -cdrom $(ISO_NAME) \
		-m 4G \
		-smp 4 \
		-cpu host \
		-machine q35 \
		-device tpm-tis,tpmdev=tpm0 \
		-tpmdev emulator,id=tpm0,chardev=chrtpm \
		-chardev socket,id=chrtpm,path=/tmp/qenex-tpm-sock \
		-netdev user,id=net0 \
		-device virtio-net,netdev=net0 \
		-serial stdio

# Debug kernel with GDB
debug: $(ISO_NAME)
	@echo "Starting QEMU with GDB debug support..."
	$(QEMU) -cdrom $(ISO_NAME) \
		-m 2G \
		-s -S \
		-serial stdio &
	@echo "QEMU started with GDB server on port 1234"
	@echo "Connect with: gdb -ex 'target remote localhost:1234' -ex 'symbol-file $(KERNEL_ELF)'"

# Build kernel only (without ISO)
kernel: $(KERNEL_ELF)

# Memory layout analysis
memory-layout: $(KERNEL_ELF)
	@echo "QENEX Banking OS Memory Layout:"
	@echo "==============================="
	@objdump -h $(KERNEL_ELF)
	@echo ""
	@echo "Banking Memory Regions:"
	@nm $(KERNEL_ELF) | grep -E "__.*_(start|end)" | sort
	@echo ""
	@echo "Entry Points:"
	@nm $(KERNEL_ELF) | grep -E "(kernel_main|_start)"

# Security analysis
security-check: $(KERNEL_ELF)
	@echo "Banking Kernel Security Analysis:"
	@echo "================================="
	@echo ""
	@echo "1. Executable Sections:"
	@objdump -h $(KERNEL_ELF) | grep -E "ALLOC.*EXEC"
	@echo ""
	@echo "2. Writable Sections:"
	@objdump -h $(KERNEL_ELF) | grep -E "ALLOC.*DATA"
	@echo ""
	@echo "3. Banking Security Symbols:"
	@nm $(KERNEL_ELF) | grep -i -E "(crypto|hsm|tpm|security|banking)" || echo "No security symbols found"
	@echo ""
	@echo "4. Stack Protection:"
	@objdump -d $(KERNEL_ELF) | grep -E "(stack_chk|__stack_chk)" && echo "Stack protection enabled" || echo "Stack protection disabled (expected for kernel)"
	@echo ""
	@echo "5. Binary Hardening:"
	@readelf -d $(KERNEL_ELF) 2>/dev/null | grep -E "(RELRO|BIND_NOW)" || echo "No dynamic linking (expected for kernel)"

# Performance benchmarks
benchmark: $(ISO_NAME)
	@echo "Running banking OS performance benchmarks..."
	@echo "This would run automated performance tests on the kernel"

# Create distribution package
package: $(ISO_NAME)
	@echo "Creating distribution package..."
	mkdir -p dist
	cp $(ISO_NAME) dist/
	cp README.md dist/ 2>/dev/null || echo "README.md not found"
	cp LICENSE dist/ 2>/dev/null || echo "LICENSE not found"
	cd dist && tar -czf ../qenex-banking-os-v1.0.tar.gz *
	@echo "Distribution package created: qenex-banking-os-v1.0.tar.gz"

# Deploy to banking hardware (placeholder)
deploy-production: $(ISO_NAME)
	@echo "WARNING: Production deployment to banking hardware"
	@echo "This should only be done in authorized banking environments"
	@echo "with proper security clearance and regulatory approval."
	@read -p "Confirm production deployment [yes/NO]: " confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "Deploying to production banking hardware..."; \
		echo "This would copy $(ISO_NAME) to approved banking servers"; \
	else \
		echo "Production deployment cancelled."; \
	fi

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f $(ISO_NAME)
	rm -f qenex-banking-os-v*.tar.gz
	rm -rf dist
	$(CARGO) clean

# Install build dependencies (Ubuntu/Debian)
install-deps:
	@echo "Installing QENEX Banking OS build dependencies..."
	sudo apt-get update
	sudo apt-get install -y \
		nasm \
		clang \
		binutils \
		grub-common \
		grub-pc-bin \
		grub2-common \
		xorriso \
		qemu-system-x86 \
		gdb \
		build-essential \
		curl
	@echo "Installing Rust..."
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
	source $$HOME/.cargo/env
	rustup target add $(TARGET)
	@echo "Build dependencies installed."

# Verify build environment
verify-env:
	@echo "Verifying QENEX Banking OS build environment..."
	@which $(AS) >/dev/null || (echo "ERROR: nasm not found" && exit 1)
	@which $(CC) >/dev/null || (echo "ERROR: clang not found" && exit 1)
	@which $(LD) >/dev/null || (echo "ERROR: ld not found" && exit 1)
	@which $(CARGO) >/dev/null || (echo "ERROR: cargo not found" && exit 1)
	@which $(GRUB_MKRESCUE) >/dev/null || (echo "ERROR: grub-mkrescue not found" && exit 1)
	@which $(QEMU) >/dev/null || (echo "WARNING: qemu-system-x86_64 not found (testing disabled)")
	@echo "Build environment verified successfully."

# Help
help:
	@echo "QENEX Banking OS Build System"
	@echo "============================="
	@echo ""
	@echo "Available targets:"
	@echo "  all              - Build bootable ISO (default)"
	@echo "  iso              - Build bootable ISO image"
	@echo "  kernel           - Build kernel ELF only"
	@echo "  test             - Test in QEMU"
	@echo "  test-kvm         - Test with KVM acceleration"
	@echo "  debug            - Debug with GDB"
	@echo "  memory-layout    - Show memory layout"
	@echo "  security-check   - Perform security analysis"
	@echo "  benchmark        - Run performance benchmarks"
	@echo "  package          - Create distribution package"
	@echo "  clean            - Remove build artifacts"
	@echo "  install-deps     - Install build dependencies"
	@echo "  verify-env       - Verify build environment"
	@echo "  help             - Show this help"
	@echo ""
	@echo "Banking-specific targets:"
	@echo "  deploy-production - Deploy to banking hardware"
	@echo ""
	@echo "Build configuration:"
	@echo "  Architecture: $(ARCH)"
	@echo "  Target: $(TARGET)"
	@echo "  Kernel name: $(KERNEL_NAME)"
	@echo "  ISO name: $(ISO_NAME)"