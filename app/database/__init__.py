# app/database/__init__.py
from .database_main import db, init_db, get_user_by_id, get_latest_water_data, insert_water_data

__all__ = ['db', 'init_db', 'get_user_by_id', 'get_latest_water_data', 'insert_water_data']

