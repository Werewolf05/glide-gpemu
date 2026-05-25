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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(gpu, model, layer_type, config)
        )
    ''')

    conn.commit()
    conn.close()


def record_layer_cost(
    gpu: str,
    model: str,
    layer_type: str,
    config: Dict[str, Any],
    compute_cost_ms: float,
    memory_cost_mb: float,
    db_path: str = LAYER_DB_PATH,
) -> None:
    """Insert or update layer profile."""

    if not os.path.exists(db_path):
        init_db(db_path)

    config_json = json.dumps(config, sort_keys=True)

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
            memory_cost_mb
        )
        VALUES (?, ?, ?, ?, ?, ?)

        ON CONFLICT(gpu, model, layer_type, config)
        DO UPDATE SET
            compute_cost_ms = excluded.compute_cost_ms,
            memory_cost_mb = excluded.memory_cost_mb,
            created_at = CURRENT_TIMESTAMP
        ''',
        (
            gpu,
            model,
            layer_type,
            config_json,
            compute_cost_ms,
            memory_cost_mb,
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

    config_json = json.dumps(config, sort_keys=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT compute_cost_ms, memory_cost_mb
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
            'memory_cost_mb': row[1]
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
            memory_cost_mb
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