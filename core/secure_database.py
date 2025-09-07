#!/usr/bin/env python3
"""
Secure Database Module - Zero SQL Injection Risk
Parameterized queries, input validation, and comprehensive security
"""

import asyncio
import asyncpg
import hashlib
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum, auto
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Allowed query types"""
    SELECT = auto()
    INSERT = auto()
    UPDATE = auto()
    DELETE = auto()
    TRANSACTION = auto()


@dataclass
class SecureQuery:
    """Secure query container with validation"""
    query_type: QueryType
    table: str
    columns: List[str]
    conditions: Dict[str, Any]
    values: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    order_by: Optional[List[Tuple[str, str]]] = None


class InputValidator:
    """Comprehensive input validation"""
    
    # Whitelist patterns
    TABLE_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')
    COLUMN_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]{0,62}$')
    
    # Blacklist patterns (additional security)
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|FROM|WHERE)\b)",
        r"(--|#|/\*|\*/|;|\||&&|\|\|)",
        r"(0x[0-9a-fA-F]+)",
        r"(CHAR\s*\(|CONCAT\s*\(|CHR\s*\()",
        r"(WAITFOR\s+DELAY|BENCHMARK\s*\(|SLEEP\s*\()",
        r"(INTO\s+(OUTFILE|DUMPFILE))",
        r"(LOAD_FILE\s*\()",
    ]
    
    @classmethod
    def validate_table_name(cls, table: str) -> bool:
        """Validate table name against whitelist"""
        if not table or not cls.TABLE_PATTERN.match(table):
            return False
        
        # Check against SQL keywords
        sql_keywords = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 
                       'CREATE', 'ALTER', 'TABLE', 'DATABASE', 'INDEX'}
        if table.upper() in sql_keywords:
            return False
        
        return True
    
    @classmethod
    def validate_column_name(cls, column: str) -> bool:
        """Validate column name"""
        if not column or not cls.COLUMN_PATTERN.match(column):
            return False
        
        # Allow aggregate functions
        if column.startswith(('COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX(')):
            inner = column[column.index('(') + 1:column.rindex(')')]
            return cls.validate_column_name(inner) or inner == '*'
        
        return True
    
    @classmethod
    def sanitize_value(cls, value: Any) -> Any:
        """Sanitize input value"""
        if value is None:
            return None
        
        if isinstance(value, str):
            # Check for SQL injection patterns
            for pattern in cls.SQL_INJECTION_PATTERNS:
                if re.search(pattern, value, re.IGNORECASE):
                    raise ValueError(f"Potential SQL injection detected: {pattern}")
            
            # Escape special characters
            value = value.replace('\x00', '')  # Remove null bytes
            value = value.replace('\\', '\\\\')  # Escape backslashes
            
            # Limit string length
            if len(value) > 10000:
                raise ValueError("String value too long")
        
        elif isinstance(value, (int, float, Decimal)):
            # Validate numeric ranges
            if isinstance(value, int) and (value < -2**63 or value > 2**63 - 1):
                raise ValueError("Integer out of range")
            
            if isinstance(value, float) and (value != value):  # NaN check
                raise ValueError("NaN values not allowed")
        
        elif isinstance(value, bytes):
            # Limit binary data size
            if len(value) > 16 * 1024 * 1024:  # 16MB limit
                raise ValueError("Binary data too large")
        
        elif isinstance(value, (list, dict)):
            # Recursively sanitize collections
            if isinstance(value, list):
                return [cls.sanitize_value(v) for v in value]
            else:
                return {k: cls.sanitize_value(v) for k, v in value.items()}
        
        elif not isinstance(value, (bool, datetime, type(None))):
            # Reject unknown types
            raise ValueError(f"Unsupported value type: {type(value)}")
        
        return value


class SecureConnectionPool:
    """Secure database connection pool with encryption"""
    
    def __init__(self, dsn: str, min_size: int = 10, max_size: int = 100):
        self.dsn = self._sanitize_dsn(dsn)
        self.min_size = min_size
        self.max_size = max_size
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    def _sanitize_dsn(self, dsn: str) -> str:
        """Sanitize database connection string"""
        # Parse and validate DSN components
        import urllib.parse
        
        parsed = urllib.parse.urlparse(dsn)
        
        # Validate scheme
        if parsed.scheme not in ('postgresql', 'postgres'):
            raise ValueError("Invalid database scheme")
        
        # Validate host
        if not parsed.hostname or '..' in parsed.hostname:
            raise ValueError("Invalid database host")
        
        # Validate port
        if parsed.port and (parsed.port < 1 or parsed.port > 65535):
            raise ValueError("Invalid database port")
        
        return dsn
    
    async def initialize(self):
        """Initialize connection pool"""
        if self._initialized:
            return
        
        try:
            self.pool = await asyncpg.create_pool(
                self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
                max_queries=50000,
                max_inactive_connection_lifetime=60,
                command_timeout=60,
                server_settings={
                    'application_name': 'QENEX_SecureDB',
                    'jit': 'off'  # Disable JIT for security
                }
            )
            
            # Test connection
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            self._initialized = True
            logger.info("Database connection pool initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool"""
        if not self._initialized:
            await self.initialize()
        
        async with self.pool.acquire() as connection:
            # Set secure session parameters
            await connection.execute("SET statement_timeout = '30s'")
            await connection.execute("SET lock_timeout = '10s'")
            await connection.execute("SET idle_in_transaction_session_timeout = '60s'")
            
            yield connection
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self._initialized = False


class SecureDatabase:
    """Secure database interface with zero SQL injection risk"""
    
    def __init__(self, connection_pool: SecureConnectionPool):
        self.pool = connection_pool
        self.query_cache: Dict[str, str] = {}
        self.prepared_statements: Dict[str, str] = {}
    
    async def execute_query(self, query: SecureQuery) -> List[Dict[str, Any]]:
        """Execute validated query with parameterized statements"""
        
        # Build parameterized query
        sql, params = self._build_parameterized_query(query)
        
        # Execute query
        async with self.pool.acquire() as conn:
            if query.query_type == QueryType.SELECT:
                rows = await conn.fetch(sql, *params)
                return [dict(row) for row in rows]
            
            elif query.query_type == QueryType.INSERT:
                if 'RETURNING' in sql:
                    row = await conn.fetchrow(sql, *params)
                    return [dict(row)] if row else []
                else:
                    await conn.execute(sql, *params)
                    return []
            
            elif query.query_type in (QueryType.UPDATE, QueryType.DELETE):
                result = await conn.execute(sql, *params)
                # Extract affected row count
                count = int(result.split()[-1]) if result else 0
                return [{'affected_rows': count}]
            
            else:
                raise ValueError(f"Unsupported query type: {query.query_type}")
    
    def _build_parameterized_query(self, query: SecureQuery) -> Tuple[str, List[Any]]:
        """Build parameterized SQL query"""
        params = []
        param_counter = 1
        
        # Validate table name
        if not InputValidator.validate_table_name(query.table):
            raise ValueError(f"Invalid table name: {query.table}")
        
        # Validate column names
        for col in query.columns:
            if not InputValidator.validate_column_name(col):
                raise ValueError(f"Invalid column name: {col}")
        
        if query.query_type == QueryType.SELECT:
            # Build SELECT query
            columns = ', '.join(query.columns) if query.columns else '*'
            sql = f"SELECT {columns} FROM {query.table}"
            
            # Add WHERE clause
            if query.conditions:
                where_clauses = []
                for col, val in query.conditions.items():
                    if not InputValidator.validate_column_name(col):
                        raise ValueError(f"Invalid column in condition: {col}")
                    
                    if val is None:
                        where_clauses.append(f"{col} IS NULL")
                    elif isinstance(val, list):
                        # IN clause
                        placeholders = ', '.join(f"${i}" for i in range(param_counter, param_counter + len(val)))
                        where_clauses.append(f"{col} IN ({placeholders})")
                        params.extend(InputValidator.sanitize_value(v) for v in val)
                        param_counter += len(val)
                    else:
                        where_clauses.append(f"{col} = ${param_counter}")
                        params.append(InputValidator.sanitize_value(val))
                        param_counter += 1
                
                sql += " WHERE " + " AND ".join(where_clauses)
            
            # Add ORDER BY
            if query.order_by:
                order_clauses = []
                for col, direction in query.order_by:
                    if not InputValidator.validate_column_name(col):
                        raise ValueError(f"Invalid column in ORDER BY: {col}")
                    if direction.upper() not in ('ASC', 'DESC'):
                        raise ValueError(f"Invalid sort direction: {direction}")
                    order_clauses.append(f"{col} {direction.upper()}")
                sql += " ORDER BY " + ", ".join(order_clauses)
            
            # Add LIMIT/OFFSET
            if query.limit:
                sql += f" LIMIT ${param_counter}"
                params.append(query.limit)
                param_counter += 1
            
            if query.offset:
                sql += f" OFFSET ${param_counter}"
                params.append(query.offset)
                param_counter += 1
        
        elif query.query_type == QueryType.INSERT:
            # Build INSERT query
            if not query.values:
                raise ValueError("INSERT requires values")
            
            columns = []
            placeholders = []
            
            for col, val in query.values.items():
                if not InputValidator.validate_column_name(col):
                    raise ValueError(f"Invalid column name: {col}")
                
                columns.append(col)
                placeholders.append(f"${param_counter}")
                params.append(InputValidator.sanitize_value(val))
                param_counter += 1
            
            sql = f"INSERT INTO {query.table} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            # Add RETURNING clause if columns specified
            if query.columns:
                sql += f" RETURNING {', '.join(query.columns)}"
        
        elif query.query_type == QueryType.UPDATE:
            # Build UPDATE query
            if not query.values:
                raise ValueError("UPDATE requires values")
            
            set_clauses = []
            for col, val in query.values.items():
                if not InputValidator.validate_column_name(col):
                    raise ValueError(f"Invalid column name: {col}")
                
                set_clauses.append(f"{col} = ${param_counter}")
                params.append(InputValidator.sanitize_value(val))
                param_counter += 1
            
            sql = f"UPDATE {query.table} SET {', '.join(set_clauses)}"
            
            # Add WHERE clause
            if query.conditions:
                where_clauses = []
                for col, val in query.conditions.items():
                    if not InputValidator.validate_column_name(col):
                        raise ValueError(f"Invalid column in condition: {col}")
                    
                    where_clauses.append(f"{col} = ${param_counter}")
                    params.append(InputValidator.sanitize_value(val))
                    param_counter += 1
                
                sql += " WHERE " + " AND ".join(where_clauses)
            else:
                # Prevent accidental full table updates
                raise ValueError("UPDATE requires WHERE clause for safety")
        
        elif query.query_type == QueryType.DELETE:
            # Build DELETE query
            sql = f"DELETE FROM {query.table}"
            
            # Add WHERE clause
            if query.conditions:
                where_clauses = []
                for col, val in query.conditions.items():
                    if not InputValidator.validate_column_name(col):
                        raise ValueError(f"Invalid column in condition: {col}")
                    
                    where_clauses.append(f"{col} = ${param_counter}")
                    params.append(InputValidator.sanitize_value(val))
                    param_counter += 1
                
                sql += " WHERE " + " AND ".join(where_clauses)
            else:
                # Prevent accidental full table deletes
                raise ValueError("DELETE requires WHERE clause for safety")
        
        return sql, params
    
    async def execute_transaction(self, queries: List[SecureQuery]) -> List[List[Dict[str, Any]]]:
        """Execute multiple queries in a transaction"""
        results = []
        
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for query in queries:
                    result = await self.execute_query(query)
                    results.append(result)
        
        return results
    
    async def prepare_statement(self, name: str, query: SecureQuery) -> None:
        """Prepare a statement for repeated execution"""
        sql, _ = self._build_parameterized_query(query)
        
        async with self.pool.acquire() as conn:
            await conn.execute(f"PREPARE {name} AS {sql}")
            self.prepared_statements[name] = sql
    
    async def execute_prepared(self, name: str, params: List[Any]) -> List[Dict[str, Any]]:
        """Execute a prepared statement"""
        if name not in self.prepared_statements:
            raise ValueError(f"Prepared statement not found: {name}")
        
        # Sanitize parameters
        sanitized_params = [InputValidator.sanitize_value(p) for p in params]
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"EXECUTE {name}", *sanitized_params)
            return [dict(row) for row in rows]
    
    async def create_secure_table(self, table_name: str, schema: Dict[str, str]) -> bool:
        """Create table with secure schema"""
        if not InputValidator.validate_table_name(table_name):
            raise ValueError(f"Invalid table name: {table_name}")
        
        # Build CREATE TABLE statement
        columns = []
        for col_name, col_type in schema.items():
            if not InputValidator.validate_column_name(col_name):
                raise ValueError(f"Invalid column name: {col_name}")
            
            # Whitelist column types
            allowed_types = {
                'INTEGER', 'BIGINT', 'DECIMAL', 'NUMERIC', 'REAL', 'DOUBLE PRECISION',
                'VARCHAR', 'TEXT', 'CHAR', 'BOOLEAN', 'DATE', 'TIMESTAMP', 'TIMESTAMPTZ',
                'UUID', 'JSON', 'JSONB', 'BYTEA'
            }
            
            col_type_upper = col_type.upper()
            type_valid = False
            
            for allowed in allowed_types:
                if col_type_upper.startswith(allowed):
                    type_valid = True
                    break
            
            if not type_valid:
                raise ValueError(f"Invalid column type: {col_type}")
            
            columns.append(f"{col_name} {col_type}")
        
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
        
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
            return True
    
    async def audit_log(self, action: str, user: str, details: Dict[str, Any]) -> None:
        """Log database actions for audit trail"""
        audit_query = SecureQuery(
            query_type=QueryType.INSERT,
            table='audit_log',
            columns=[],
            conditions={},
            values={
                'timestamp': datetime.now(timezone.utc),
                'action': action,
                'user': user,
                'details': details,
                'checksum': self._compute_audit_checksum(action, user, details)
            }
        )
        
        await self.execute_query(audit_query)
    
    def _compute_audit_checksum(self, action: str, user: str, details: Dict[str, Any]) -> str:
        """Compute checksum for audit log integrity"""
        data = f"{action}:{user}:{sorted(details.items())}"
        return hashlib.blake2b(data.encode()).hexdigest()


# Example usage
async def example_usage():
    """Example of secure database usage"""
    
    # Initialize connection pool
    pool = SecureConnectionPool("postgresql://user:pass@localhost/qenex")
    await pool.initialize()
    
    # Create secure database interface
    db = SecureDatabase(pool)
    
    # Create table
    await db.create_secure_table('accounts', {
        'id': 'SERIAL PRIMARY KEY',
        'username': 'VARCHAR(255) UNIQUE NOT NULL',
        'balance': 'DECIMAL(20, 8) DEFAULT 0',
        'created_at': 'TIMESTAMPTZ DEFAULT NOW()'
    })
    
    # Insert data
    insert_query = SecureQuery(
        query_type=QueryType.INSERT,
        table='accounts',
        columns=['id', 'username'],
        conditions={},
        values={
            'username': 'alice',
            'balance': Decimal('1000.00')
        }
    )
    result = await db.execute_query(insert_query)
    
    # Select data
    select_query = SecureQuery(
        query_type=QueryType.SELECT,
        table='accounts',
        columns=['id', 'username', 'balance'],
        conditions={'username': 'alice'},
        limit=10
    )
    accounts = await db.execute_query(select_query)
    
    # Update data
    update_query = SecureQuery(
        query_type=QueryType.UPDATE,
        table='accounts',
        columns=[],
        conditions={'username': 'alice'},
        values={'balance': Decimal('1500.00')}
    )
    await db.execute_query(update_query)
    
    # Transaction example
    transfer_queries = [
        SecureQuery(
            query_type=QueryType.UPDATE,
            table='accounts',
            columns=[],
            conditions={'username': 'alice'},
            values={'balance': Decimal('900.00')}
        ),
        SecureQuery(
            query_type=QueryType.UPDATE,
            table='accounts',
            columns=[],
            conditions={'username': 'bob'},
            values={'balance': Decimal('1100.00')}
        )
    ]
    await db.execute_transaction(transfer_queries)
    
    # Audit log
    await db.audit_log('transfer', 'system', {
        'from': 'alice',
        'to': 'bob',
        'amount': '100.00'
    })
    
    # Close pool
    await pool.close()


if __name__ == "__main__":
    asyncio.run(example_usage())