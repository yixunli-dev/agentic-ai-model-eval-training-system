from pathlib import Path

DEFAULT_DATABASE_PATH = Path("outputs/experiments.db")


def get_database_path(database_path=None):
    if database_path is None:
        return DEFAULT_DATABASE_PATH
    return Path(database_path)
