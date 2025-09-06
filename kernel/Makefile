obj-m += qenex_kernel.o

all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

install:
	sudo insmod qenex_kernel.ko

remove:
	sudo rmmod qenex_kernel

test:
	@echo "Testing QENEX kernel module..."
	@cat /dev/qenex_banking