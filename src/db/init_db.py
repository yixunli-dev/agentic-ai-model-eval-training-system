import sqlite3
from pathlib import Path

from src.db.database import get_database_path


def init_database(database_path=None):
    path = get_database_path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                task TEXT NOT NULL,
                model_type TEXT NOT NULL,
                status TEXT NOT NULL,
                config_path TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                metrics_json TEXT,
                report_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.commit()
    return path
