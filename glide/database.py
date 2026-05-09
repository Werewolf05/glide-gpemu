"""
Layer cost database for GLIDE profiler.

Stores profiled layer execution times and memory usage indexed by
layer type and configuration.
"""

import json
import os
import sqlite3
from pathlib import Path
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
            layer_type TEXT NOT NULL,
            config TEXT NOT NULL,
            compute_cost_ms REAL,
            memory_cost_mb REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(layer_type, config)
        )
    ''')

    conn.commit()
    conn.close()


def record_layer_cost(
    layer_type: str,
    config: Dict[str, Any],
    compute_cost_ms: float,
    memory_cost_mb: float,
    db_path: str = LAYER_DB_PATH,
) -> None:
    """Record or update a layer's cost profile.
    
    Args:
        layer_type: Name of the layer type (e.g., 'Conv2d', 'Linear')
        config: Configuration dict for the layer
        compute_cost_ms: Average compute time in milliseconds
        memory_cost_mb: Average memory usage in MB
        db_path: Path to the database file
    """
    # Ensure database exists
    if not os.path.exists(db_path):
        init_db(db_path)

    config_json = json.dumps(config, sort_keys=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert or update
    cursor.execute('''
        INSERT INTO layers (layer_type, config, compute_cost_ms, memory_cost_mb)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(layer_type, config) DO UPDATE SET
            compute_cost_ms = excluded.compute_cost_ms,
            memory_cost_mb = excluded.memory_cost_mb,
            created_at = CURRENT_TIMESTAMP
    ''', (layer_type, config_json, compute_cost_ms, memory_cost_mb))

    conn.commit()
    conn.close()


def query_layer_cost(
    layer_type: str,
    config: Dict[str, Any],
    db_path: str = LAYER_DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Query a layer's profiled cost.
    
    Args:
        layer_type: Name of the layer type
        config: Configuration dict for the layer
        db_path: Path to the database file

    Returns:
        Dict with compute_cost_ms and memory_cost_mb, or None if not found
    """
    if not os.path.exists(db_path):
        return None

    config_json = json.dumps(config, sort_keys=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT compute_cost_ms, memory_cost_mb FROM layers
        WHERE layer_type = ? AND config = ?
    ''', (layer_type, config_json))

    row = cursor.fetchone()
    conn.close()

    if row:
        return {'compute_cost_ms': row[0], 'memory_cost_mb': row[1]}
    return None


def get_slowest_layers(
    limit: int = 5,
    db_path: str = LAYER_DB_PATH,
) -> list:
    """Get the slowest layers by compute time.
    
    Args:
        limit: Number of results to return
        db_path: Path to the database file

    Returns:
        List of dicts with layer_type, config, compute_cost_ms, memory_cost_mb
    """
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT layer_type, config, compute_cost_ms, memory_cost_mb
        FROM layers
        ORDER BY compute_cost_ms DESC
        LIMIT ?
    ''', (limit,))

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
