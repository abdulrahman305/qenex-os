/*
 * QENEX Banking OS Kernel Module
 * Real Linux kernel module for financial transaction processing
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#include <linux/time.h>
#include <linux/crypto.h>
#include <linux/scatterlist.h>
#include <linux/random.h>

#define DEVICE_NAME "qenex_banking"
#define CLASS_NAME "qenex"
#define QENEX_MAGIC 0xBANK

MODULE_LICENSE("GPL");
MODULE_AUTHOR("QENEX Team");
MODULE_DESCRIPTION("QENEX Banking OS Kernel Module");
MODULE_VERSION("1.0");

/* Transaction structure */
struct qenex_transaction {
    u64 id;
    u64 from_account;
    u64 to_account;
    s64 amount;  /* in cents to avoid floating point */
    u32 currency;
    u32 status;
    struct timespec64 timestamp;
    struct list_head list;
};

/* Account structure */
struct qenex_account {
    u64 account_number;
    s64 balance;  /* in cents */
    u32 currency;
    u32 status;
    spinlock_t lock;
    struct list_head transactions;
    struct list_head list;
};

/* Global variables */
static int major_number;
static struct class* qenex_class = NULL;
static struct device* qenex_device = NULL;
static struct cdev qenex_cdev;

static DEFINE_MUTEX(accounts_mutex);
static LIST_HEAD(accounts_list);
static atomic64_t next_account_number = ATOMIC64_INIT(1000000000);
static atomic64_t next_transaction_id = ATOMIC64_INIT(1);

/* IOCTL commands */
#define QENEX_CREATE_ACCOUNT    _IOR(QENEX_MAGIC, 1, struct qenex_account_request)
#define QENEX_GET_BALANCE       _IOR(QENEX_MAGIC, 2, struct qenex_balance_request)
#define QENEX_TRANSFER          _IOW(QENEX_MAGIC, 3, struct qenex_transfer_request)
#define QENEX_GET_TRANSACTION   _IOR(QENEX_MAGIC, 4, struct qenex_transaction_request)

/* Request structures */
struct qenex_account_request {
    s64 initial_balance;
    u32 currency;
    u64 account_number;  /* returned */
};

struct qenex_balance_request {
    u64 account_number;
    s64 balance;  /* returned */
};

struct qenex_transfer_request {
    u64 from_account;
    u64 to_account;
    s64 amount;
    u32 currency;
    u64 transaction_id;  /* returned */
};

struct qenex_transaction_request {
    u64 transaction_id;
    u32 status;  /* returned */
};

/* Find account by number */
static struct qenex_account* find_account(u64 account_number)
{
    struct qenex_account *account;
    
    list_for_each_entry(account, &accounts_list, list) {
        if (account->account_number == account_number)
            return account;
    }
    return NULL;
}

/* Create new account */
static long create_account(struct qenex_account_request __user *req)
{
    struct qenex_account_request kreq;
    struct qenex_account *account;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    account = kzalloc(sizeof(*account), GFP_KERNEL);
    if (!account)
        return -ENOMEM;
    
    account->account_number = atomic64_inc_return(&next_account_number);
    account->balance = kreq.initial_balance;
    account->currency = kreq.currency;
    account->status = 1;  /* active */
    spin_lock_init(&account->lock);
    INIT_LIST_HEAD(&account->transactions);
    
    mutex_lock(&accounts_mutex);
    list_add(&account->list, &accounts_list);
    mutex_unlock(&accounts_mutex);
    
    kreq.account_number = account->account_number;
    
    if (copy_to_user(req, &kreq, sizeof(kreq)))
        return -EFAULT;
    
    printk(KERN_INFO "QENEX: Account %llu created with balance %lld\n",
           account->account_number, account->balance);
    
    return 0;
}

/* Get account balance */
static long get_balance(struct qenex_balance_request __user *req)
{
    struct qenex_balance_request kreq;
    struct qenex_account *account;
    unsigned long flags;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    mutex_lock(&accounts_mutex);
    account = find_account(kreq.account_number);
    if (!account) {
        mutex_unlock(&accounts_mutex);
        return -ENOENT;
    }
    
    spin_lock_irqsave(&account->lock, flags);
    kreq.balance = account->balance;
    spin_unlock_irqrestore(&account->lock, flags);
    mutex_unlock(&accounts_mutex);
    
    if (copy_to_user(req, &kreq, sizeof(kreq)))
        return -EFAULT;
    
    return 0;
}

/* Process transfer between accounts */
static long process_transfer(struct qenex_transfer_request __user *req)
{
    struct qenex_transfer_request kreq;
    struct qenex_account *from_account, *to_account;
    struct qenex_transaction *transaction;
    unsigned long flags;
    int ret = 0;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    if (kreq.amount <= 0)
        return -EINVAL;
    
    transaction = kzalloc(sizeof(*transaction), GFP_KERNEL);
    if (!transaction)
        return -ENOMEM;
    
    mutex_lock(&accounts_mutex);
    
    from_account = find_account(kreq.from_account);
    to_account = find_account(kreq.to_account);
    
    if (!from_account || !to_account) {
        ret = -ENOENT;
        goto out;
    }
    
    /* Lock accounts in order to prevent deadlock */
    if (from_account->account_number < to_account->account_number) {
        spin_lock_irqsave(&from_account->lock, flags);
        spin_lock(&to_account->lock);
    } else {
        spin_lock_irqsave(&to_account->lock, flags);
        spin_lock(&from_account->lock);
    }
    
    /* Check sufficient balance */
    if (from_account->balance < kreq.amount) {
        ret = -EINVAL;
        goto unlock;
    }
    
    /* Perform transfer */
    from_account->balance -= kreq.amount;
    to_account->balance += kreq.amount;
    
    /* Record transaction */
    transaction->id = atomic64_inc_return(&next_transaction_id);
    transaction->from_account = kreq.from_account;
    transaction->to_account = kreq.to_account;
    transaction->amount = kreq.amount;
    transaction->currency = kreq.currency;
    transaction->status = 1;  /* completed */
    ktime_get_real_ts64(&transaction->timestamp);
    
    list_add(&transaction->list, &from_account->transactions);
    
    kreq.transaction_id = transaction->id;
    
    printk(KERN_INFO "QENEX: Transaction %llu: %lld from %llu to %llu\n",
           transaction->id, kreq.amount, kreq.from_account, kreq.to_account);
    
unlock:
    if (from_account->account_number < to_account->account_number) {
        spin_unlock(&to_account->lock);
        spin_unlock_irqrestore(&from_account->lock, flags);
    } else {
        spin_unlock(&from_account->lock);
        spin_unlock_irqrestore(&to_account->lock, flags);
    }
    
out:
    mutex_unlock(&accounts_mutex);
    
    if (ret) {
        kfree(transaction);
        return ret;
    }
    
    if (copy_to_user(req, &kreq, sizeof(kreq)))
        return -EFAULT;
    
    return 0;
}

/* Device file operations */
static int qenex_open(struct inode *inodep, struct file *filep)
{
    printk(KERN_INFO "QENEX: Device opened\n");
    return 0;
}

static int qenex_release(struct inode *inodep, struct file *filep)
{
    printk(KERN_INFO "QENEX: Device closed\n");
    return 0;
}

static ssize_t qenex_read(struct file *filep, char __user *buffer, size_t len, loff_t *offset)
{
    const char *message = "QENEX Banking OS Kernel Module v1.0\n";
    size_t message_len = strlen(message);
    
    if (*offset >= message_len)
        return 0;
    
    if (len > message_len - *offset)
        len = message_len - *offset;
    
    if (copy_to_user(buffer, message + *offset, len))
        return -EFAULT;
    
    *offset += len;
    return len;
}

static long qenex_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    switch (cmd) {
    case QENEX_CREATE_ACCOUNT:
        return create_account((struct qenex_account_request __user *)arg);
    case QENEX_GET_BALANCE:
        return get_balance((struct qenex_balance_request __user *)arg);
    case QENEX_TRANSFER:
        return process_transfer((struct qenex_transfer_request __user *)arg);
    default:
        return -ENOTTY;
    }
}

static struct file_operations fops = {
    .open = qenex_open,
    .read = qenex_read,
    .unlocked_ioctl = qenex_ioctl,
    .release = qenex_release,
};

/* Module initialization */
static int __init qenex_init(void)
{
    printk(KERN_INFO "QENEX: Initializing Banking OS Kernel Module\n");
    
    /* Register character device */
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        printk(KERN_ALERT "QENEX: Failed to register major number\n");
        return major_number;
    }
    
    /* Register device class */
    qenex_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(qenex_class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "QENEX: Failed to register device class\n");
        return PTR_ERR(qenex_class);
    }
    
    /* Register device driver */
    qenex_device = device_create(qenex_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(qenex_device)) {
        class_destroy(qenex_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "QENEX: Failed to create device\n");
        return PTR_ERR(qenex_device);
    }
    
    printk(KERN_INFO "QENEX: Banking OS Kernel Module loaded (major %d)\n", major_number);
    return 0;
}

/* Module cleanup */
static void __exit qenex_exit(void)
{
    struct qenex_account *account, *tmp_account;
    struct qenex_transaction *transaction, *tmp_transaction;
    
    /* Clean up accounts and transactions */
    mutex_lock(&accounts_mutex);
    list_for_each_entry_safe(account, tmp_account, &accounts_list, list) {
        list_for_each_entry_safe(transaction, tmp_transaction, &account->transactions, list) {
            list_del(&transaction->list);
            kfree(transaction);
        }
        list_del(&account->list);
        kfree(account);
    }
    mutex_unlock(&accounts_mutex);
    
    device_destroy(qenex_class, MKDEV(major_number, 0));
    class_unregister(qenex_class);
    class_destroy(qenex_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    
    printk(KERN_INFO "QENEX: Banking OS Kernel Module unloaded\n");
}

module_init(qenex_init);
module_exit(qenex_exit);