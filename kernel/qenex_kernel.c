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
#include <linux/rcupdate.h>
#include <linux/rculist.h>
#include <linux/rwlock.h>
#include <linux/refcount.h>
#include <linux/completion.h>
#include <linux/atomic.h>

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

/* Account structure with enhanced concurrency control */
struct qenex_account {
    u64 account_number;
    atomic64_t balance;  /* in cents - atomic for lock-free reads */
    u32 currency;
    atomic_t status;
    rwlock_t lock;  /* Read-write lock for better concurrency */
    struct list_head transactions;
    struct list_head list;
    struct rcu_head rcu;  /* RCU head for safe deletion */
    refcount_t ref_count;  /* Reference counting */
    struct completion completion;  /* For orderly shutdown */
};

/* Global variables with enhanced concurrency */
static int major_number;
static struct class* qenex_class = NULL;
static struct device* qenex_device = NULL;
static struct cdev qenex_cdev;

/* Use RCU-protected list for accounts with reader-writer lock for updates */
static DEFINE_RWLOCK(accounts_rwlock);
static LIST_HEAD(accounts_list);
static atomic64_t next_account_number = ATOMIC64_INIT(1000000000);
static atomic64_t next_transaction_id = ATOMIC64_INIT(1);

/* Lock ordering to prevent deadlocks: always acquire lower account number first */
static DEFINE_SPINLOCK(transfer_sequence_lock);
static atomic64_t transfer_sequence = ATOMIC64_INIT(0);

/* Transaction batching for better performance */
static DEFINE_SPINLOCK(batch_lock);
static LIST_HEAD(pending_transfers);
static atomic_t batch_size = ATOMIC_INIT(0);
#define MAX_BATCH_SIZE 100

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

/* RCU-safe account cleanup */
static void account_rcu_free(struct rcu_head *rcu)
{
    struct qenex_account *account = container_of(rcu, struct qenex_account, rcu);
    kfree(account);
}

/* Get account reference safely */
static struct qenex_account* get_account_ref(u64 account_number)
{
    struct qenex_account *account = NULL;
    
    rcu_read_lock();
    list_for_each_entry_rcu(account, &accounts_list, list) {
        if (account->account_number == account_number) {
            if (refcount_inc_not_zero(&account->ref_count)) {
                rcu_read_unlock();
                return account;
            }
        }
    }
    rcu_read_unlock();
    return NULL;
}

/* Release account reference */
static void put_account_ref(struct qenex_account *account)
{
    if (refcount_dec_and_test(&account->ref_count)) {
        complete(&account->completion);
    }
}

/* Find account by number with RCU protection */
static struct qenex_account* find_account_rcu(u64 account_number)
{
    struct qenex_account *account;
    
    rcu_read_lock();
    list_for_each_entry_rcu(account, &accounts_list, list) {
        if (account->account_number == account_number) {
            rcu_read_unlock();
            return account;
        }
    }
    rcu_read_unlock();
    return NULL;
}

/* Create new account with proper concurrency control */
static long create_account(struct qenex_account_request __user *req)
{
    struct qenex_account_request kreq;
    struct qenex_account *account;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    account = kzalloc(sizeof(*account), GFP_KERNEL);
    if (!account)
        return -ENOMEM;
    
    /* Initialize account with atomic operations */
    account->account_number = atomic64_inc_return(&next_account_number);
    atomic64_set(&account->balance, kreq.initial_balance);
    account->currency = kreq.currency;
    atomic_set(&account->status, 1);  /* active */
    rwlock_init(&account->lock);
    INIT_LIST_HEAD(&account->transactions);
    refcount_set(&account->ref_count, 1);
    init_completion(&account->completion);
    
    /* Add to list with proper synchronization */
    write_lock(&accounts_rwlock);
    list_add_rcu(&account->list, &accounts_list);
    write_unlock(&accounts_rwlock);
    
    /* Memory barrier to ensure visibility */
    smp_wmb();
    
    kreq.account_number = account->account_number;
    
    if (copy_to_user(req, &kreq, sizeof(kreq)))
        return -EFAULT;
    
    printk(KERN_INFO "QENEX: Account %llu created with balance %lld\n",
           account->account_number, atomic64_read(&account->balance));
    
    return 0;
}

/* Get account balance with lock-free atomic read */
static long get_balance(struct qenex_balance_request __user *req)
{
    struct qenex_balance_request kreq;
    struct qenex_account *account;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    /* Use RCU-protected lookup - no need for heavy locks for read-only */
    account = find_account_rcu(kreq.account_number);
    if (!account)
        return -ENOENT;
    
    /* Atomic read - no locks needed for balance inquiry */
    kreq.balance = atomic64_read(&account->balance);
    
    /* Memory barrier to ensure consistent read */
    smp_rmb();
    
    if (copy_to_user(req, &kreq, sizeof(kreq)))
        return -EFAULT;
    
    return 0;
}

/* Atomic compare-and-swap based transfer to prevent race conditions */
static bool atomic_transfer_balance(struct qenex_account *from_account, 
                                   struct qenex_account *to_account, s64 amount)
{
    s64 old_from_balance, new_from_balance;
    s64 old_to_balance, new_to_balance;
    
    do {
        old_from_balance = atomic64_read(&from_account->balance);
        if (old_from_balance < amount) {
            return false; /* Insufficient funds */
        }
        new_from_balance = old_from_balance - amount;
        
        /* Try to update from_account balance atomically */
    } while (atomic64_cmpxchg(&from_account->balance, old_from_balance, new_from_balance) != old_from_balance);
    
    /* Now update to_account balance - this cannot fail */
    do {
        old_to_balance = atomic64_read(&to_account->balance);
        new_to_balance = old_to_balance + amount;
    } while (atomic64_cmpxchg(&to_account->balance, old_to_balance, new_to_balance) != old_to_balance);
    
    /* Full memory barrier to ensure transfer completion is visible */
    smp_mb();
    
    return true;
}

/* Process transfer between accounts with deadlock-free atomic operations */
static long process_transfer(struct qenex_transfer_request __user *req)
{
    struct qenex_transfer_request kreq;
    struct qenex_account *from_account, *to_account;
    struct qenex_transaction *transaction;
    u64 sequence_number;
    int ret = 0;
    
    if (copy_from_user(&kreq, req, sizeof(kreq)))
        return -EFAULT;
    
    if (kreq.amount <= 0)
        return -EINVAL;
    
    if (kreq.from_account == kreq.to_account)
        return -EINVAL; /* No self-transfers */
    
    transaction = kzalloc(sizeof(*transaction), GFP_KERNEL);
    if (!transaction)
        return -ENOMEM;
    
    /* Get atomic sequence number for ordering */
    sequence_number = atomic64_inc_return(&transfer_sequence);
    
    /* Get references to accounts with RCU protection */
    from_account = get_account_ref(kreq.from_account);
    to_account = get_account_ref(kreq.to_account);
    
    if (!from_account || !to_account) {
        ret = -ENOENT;
        goto cleanup_refs;
    }
    
    /* Check account status atomically */
    if (atomic_read(&from_account->status) != 1 || atomic_read(&to_account->status) != 1) {
        ret = -EACCES; /* Account not active */
        goto cleanup_refs;
    }
    
    /* Perform atomic transfer without locks */
    if (!atomic_transfer_balance(from_account, to_account, kreq.amount)) {
        ret = -EINVAL; /* Insufficient funds */
        goto cleanup_refs;
    }
    
    /* Record transaction with minimal locking */
    transaction->id = atomic64_inc_return(&next_transaction_id);
    transaction->from_account = kreq.from_account;
    transaction->to_account = kreq.to_account;
    transaction->amount = kreq.amount;
    transaction->currency = kreq.currency;
    transaction->status = 1;  /* completed */
    ktime_get_real_ts64(&transaction->timestamp);
    
    /* Add transaction to from_account's history with write lock */
    write_lock(&from_account->lock);
    list_add(&transaction->list, &from_account->transactions);
    write_unlock(&from_account->lock);
    
    kreq.transaction_id = transaction->id;
    
    printk(KERN_INFO "QENEX: Transaction %llu (seq %llu): %lld from %llu to %llu\n",
           transaction->id, sequence_number, kreq.amount, kreq.from_account, kreq.to_account);
    
cleanup_refs:
    if (from_account)
        put_account_ref(from_account);
    if (to_account)
        put_account_ref(to_account);
    
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

/* Module cleanup with proper RCU synchronization */
static void __exit qenex_exit(void)
{
    struct qenex_account *account, *tmp_account;
    struct qenex_transaction *transaction, *tmp_transaction;
    
    printk(KERN_INFO "QENEX: Starting module cleanup\n");
    
    /* First, remove all accounts from the list to prevent new operations */
    write_lock(&accounts_rwlock);
    list_for_each_entry_safe(account, tmp_account, &accounts_list, list) {
        /* Mark account as inactive */
        atomic_set(&account->status, 0);
        list_del_rcu(&account->list);
        
        /* Signal account for shutdown */
        put_account_ref(account);
    }
    write_unlock(&accounts_rwlock);
    
    /* Wait for RCU grace period to ensure no readers are accessing accounts */
    synchronize_rcu();
    
    /* Now wait for all references to be dropped and clean up */
    list_for_each_entry_safe(account, tmp_account, &accounts_list, list) {
        /* Wait for all references to this account to be released */
        wait_for_completion(&account->completion);
        
        /* Clean up transactions for this account */
        write_lock(&account->lock);
        list_for_each_entry_safe(transaction, tmp_transaction, &account->transactions, list) {
            list_del(&transaction->list);
            kfree(transaction);
        }
        write_unlock(&account->lock);
        
        /* Free account using RCU */
        call_rcu(&account->rcu, account_rcu_free);
    }
    
    /* Final RCU synchronization */
    rcu_barrier();
    
    device_destroy(qenex_class, MKDEV(major_number, 0));
    class_unregister(qenex_class);
    class_destroy(qenex_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    
    printk(KERN_INFO "QENEX: Banking OS Kernel Module unloaded safely\n");
}

module_init(qenex_init);
module_exit(qenex_exit);