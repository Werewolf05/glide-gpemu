"""
Layer cost database for GLIDE profiler.

Stores profiled layer execution times and memory usage indexed by
GPU, model, layer type, and configuration.
"""

import json
import os
import sqlite3
from typing import Any, Dict, Optional


# Get GLIDE directory dynamically
GLIDE_DIR = os.path.dirname(os.path.abspath(__file__))
LAYER_DB_PATH = os.path.join(GLIDE_DIR, 'layer_db.sqlite')


def init_db(db_path: str = LAYER_DB_PATH) -> None:
    """Initialize the layer cost database schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create layers table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS layers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gpu TEXT NOT NULL,
            model TEXT NOT NULL,
            layer_type TEXT NOT NULL,
            config TEXT NOT NULL,
            compute_cost_ms REAL,
            memory_cost_mb REAL,
            parallelism_degree INTEGER NOT NULL DEFAULT 1,
            input_shape TEXT,
            batch_size INTEGER,
            precision TEXT,
            warmup_runs INTEGER,
            measured_runs INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(gpu, model, layer_type, config)
        )
    ''')
    columns = {row[1] for row in cursor.execute('PRAGMA table_info(layers)')}
    migrations = {
        'parallelism_degree': 'INTEGER NOT NULL DEFAULT 1',
        'input_shape': 'TEXT',
        'batch_size': 'INTEGER',
        'precision': 'TEXT',
        'warmup_runs': 'INTEGER',
        'measured_runs': 'INTEGER',
    }
    for column, definition in migrations.items():
        if column not in columns:
            cursor.execute(f'ALTER TABLE layers ADD COLUMN {column} {definition}')

    conn.commit()
    conn.close()


def estimate_parallelism(layer_type: str, config: Dict[str, Any]) -> int:
    if layer_type == 'Conv2d':
        return min(32, int(config.get('out', 1)))
    if layer_type == 'Linear':
        return min(32, int(config.get('out', 1)) // 32 + 1)
    return 1


def record_layer_cost(
    gpu: str,
    model: str,
    layer_type: str,
    config: Dict[str, Any],
    compute_cost_ms: float,
    memory_cost_mb: float,
    parallelism_degree: Optional[int] = None,
    input_shape: Optional[list] = None,
    batch_size: Optional[int] = None,
    precision: Optional[str] = None,
    warmup_runs: Optional[int] = None,
    measured_runs: Optional[int] = None,
    db_path: str = LAYER_DB_PATH,
) -> None:
    """Insert or update layer profile."""

    if not os.path.exists(db_path):
        init_db(db_path)

    config_json = json.dumps(config, sort_keys=True)
    parallelism = parallelism_degree or estimate_parallelism(layer_type, config)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
        INSERT INTO layers (
            gpu,
            model,
            layer_type,
            config,
            compute_cost_ms,
            memory_cost_mb,
            parallelism_degree,
            input_shape,
            batch_size,
            precision,
            warmup_runs,
            measured_runs
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(gpu, model, layer_type, config)
        DO UPDATE SET
            compute_cost_ms = excluded.compute_cost_ms,
            memory_cost_mb = excluded.memory_cost_mb,
            parallelism_degree = excluded.parallelism_degree,
            input_shape = excluded.input_shape,
            batch_size = excluded.batch_size,
            precision = excluded.precision,
            warmup_runs = excluded.warmup_runs,
            measured_runs = excluded.measured_runs,
            created_at = CURRENT_TIMESTAMP
        ''',
        (
            gpu,
            model,
            layer_type,
            config_json,
            compute_cost_ms,
            memory_cost_mb,
            parallelism,
            json.dumps(input_shape) if input_shape is not None else None,
            batch_size,
            precision,
            warmup_runs,
            measured_runs,
        ),
    )

    conn.commit()
    conn.close()


def query_layer_cost(
    gpu: str,
    model: str,
    layer_type: str,
    config: Dict[str, Any],
    db_path: str = LAYER_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Query a layer's profiled cost."""

    if not os.path.exists(db_path):
        return None
    init_db(db_path)

    config_json = json.dumps(config, sort_keys=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT compute_cost_ms, memory_cost_mb, parallelism_degree,
               input_shape, batch_size, precision, warmup_runs, measured_runs
        FROM layers
        WHERE gpu = ?
        AND model = ?
        AND layer_type = ?
        AND config = ?
    ''', (
        gpu,
        model,
        layer_type,
        config_json
    ))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {
            'compute_cost_ms': row[0],
            'memory_cost_mb': row[1],
            'parallelism_degree': row[2],
            'input_shape': json.loads(row[3]) if row[3] else None,
            'batch_size': row[4],
            'precision': row[5],
            'warmup_runs': row[6],
            'measured_runs': row[7],
        }

    return None


def get_layer_cost(
    gpu: str,
    model: str,
    layer_type: str,
    config: Dict[str, Any],
    db_path: str = LAYER_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible alias for query_layer_cost."""
    return query_layer_cost(
        gpu,
        model,
        layer_type,
        config,
        db_path
    )


def get_slowest_layers(
    gpu: str,
    model: str,
    limit: int = 5,
    db_path: str = LAYER_DB_PATH,
) -> list:
    """Get the slowest layers by compute time."""

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT
            layer_type,
            config,
            compute_cost_ms,
            memory_cost_mb,
            parallelism_degree
        FROM layers
        WHERE gpu = ?
        AND model = ?
        ORDER BY compute_cost_ms DESC
        LIMIT ?
    ''', (
        gpu,
        model,
        limit
    ))

    rows = cursor.fetchall()
    conn.close()

    result = []

    for row in rows:
        result.append({
            'layer_type': row[0],
            'config': json.loads(row[1]),
            'compute_cost_ms': row[2],
            'memory_cost_mb': row[3],
            'parallelism_degree': row[4],
        })

    return result
# Add this to glide/database.py

def get_all_profiles(
    db_path: str = LAYER_DB_PATH,
) -> list:
    """Return all profiled GPU/model combinations."""

    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        '''
        SELECT DISTINCT gpu, model
        FROM layers
        ORDER BY gpu, model
        '''
    )

    rows = cursor.fetchall()

    conn.close()

    return [
        {
            'gpu': row[0],
            'model': row[1],
        }
        for row in rows
    ]