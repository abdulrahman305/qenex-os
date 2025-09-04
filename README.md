# QENEX Core System

Minimalist Python framework for building robust applications.

## Installation

```bash
pip install -r requirements.txt
python core.py
```

## Features

- **Database**: SQLite with connection management
- **Authentication**: Session-based auth with secure password hashing
- **Caching**: TTL-based cache with automatic cleanup
- **Monitoring**: Metrics collection and logging
- **API**: Simple routing with authentication
- **Workers**: Background task processing

## Quick Start

```python
from core import Core

# Initialize system
core = Core()

# Create user
user_id = core.auth.create_user('username', 'password')

# Authenticate
session = core.auth.authenticate('username', 'password')

# Run system
core.run()
```

## API Endpoints

- `GET /health` - System health check
- `POST /auth/register` - User registration
- `POST /auth/login` - User authentication
- `GET /metrics` - System metrics

## Configuration

Edit configuration in `core.py`:

```python
CONFIG = {
    'db_path': Path('data/core.db'),
    'log_path': Path('logs/core.log'),
    'cache_ttl': 300,
    'max_workers': 10,
    'api_timeout': 30,
}
```

## License

MIT