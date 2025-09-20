# database/database_main.py
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from contextlib import contextmanager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class DatabaseConnection:
    """Database connection manager for TimescaleDB/PostgreSQL"""
    
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'water_iot'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            # Set autocommit to True to avoid transaction issues
            conn.autocommit = True
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    @contextmanager
    def get_cursor(self, cursor_factory=None):
        """Context manager for database cursor"""
        conn = psycopg2.connect(**self.connection_params)
        conn.autocommit = True  # Use autocommit mode to avoid transaction issues
        cursor = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cursor
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def test_connection(self):
        """Test the database connection"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()
                    logger.info(f"Database connection successful: {version[0]}")
                    return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

# Create a global instance
db = DatabaseConnection()

def init_db():
    """Initialize database with required tables"""
    try:
        # First, check if we need to recover from a failed transaction
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Try to rollback any existing transaction
                try:
                    cur.execute("ROLLBACK;")
                    logger.info("Rolled back any existing transactions")
                except:
                    pass  # Ignore if no transaction to rollback
                
                # Now proceed with table creation
                # Create users table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS public.users (
                        user_id SERIAL PRIMARY KEY,
                        username VARCHAR(50) NOT NULL UNIQUE,
                        user_type VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("Users table checked/created")
                
                # Create IoT data table with alternative location storage
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS iot_water_data (
                        time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        device_id VARCHAR(50) NOT NULL,
                        ph_value FLOAT,
                        turbidity FLOAT,
                        temperature FLOAT,
                        tds_value FLOAT,
                        latitude FLOAT,
                        longitude FLOAT,
                        status VARCHAR(20) DEFAULT 'active',
                        CONSTRAINT valid_ph CHECK (ph_value >= 0 AND ph_value <= 14),
                        CONSTRAINT valid_temperature CHECK (temperature >= -50 AND temperature <= 100)
                    )
                """)
                logger.info("IoT water data table checked/created")
                
                # Try to convert to hypertable (for TimescaleDB)
                try:
                    cur.execute("""
                        SELECT create_hypertable('iot_water_data', 'time', if_not_exists => TRUE)
                    """)
                    logger.info("TimescaleDB hypertable created")
                except psycopg2.Error as e:
                    logger.warning(f"TimescaleDB extension not available: {e}")
                
                # Create index for better query performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_iot_device_time 
                    ON iot_water_data (device_id, time DESC)
                """)
                logger.info("Index checked/created")
                
                # Create water quality alerts table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS water_quality_alerts (
                        alert_id SERIAL PRIMARY KEY,
                        device_id VARCHAR(50) NOT NULL,
                        parameter VARCHAR(20) NOT NULL,
                        value FLOAT NOT NULL,
                        threshold FLOAT NOT NULL,
                        severity VARCHAR(10) CHECK (severity IN ('low', 'medium', 'high')),
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        resolved_at TIMESTAMPTZ,
                        is_resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                logger.info("Alerts table checked/created")
                
                # Insert a sample user if table is empty
                cur.execute("SELECT COUNT(*) FROM public.users")
                if cur.fetchone()[0] == 0:
                    cur.execute("""
                        INSERT INTO public.users (username, user_type) 
                        VALUES (%s, %s)
                    """, ('Kishan R0', 'admin'))
                    logger.info("Sample user inserted")
                
                logger.info("Database initialized successfully")
                
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        # Try to recover by rolling back
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("ROLLBACK;")
        except:
            pass
        raise

def drop_and_recreate_tables():
    """Drop and recreate all tables (for development only)"""
    try:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                # Drop tables in correct order to avoid foreign key constraints
                cur.execute("DROP TABLE IF EXISTS water_quality_alerts CASCADE;")
                cur.execute("DROP TABLE IF EXISTS iot_water_data CASCADE;")
                cur.execute("DROP TABLE IF EXISTS public.users CASCADE;")
                logger.info("Tables dropped successfully")
        
        # Now recreate them
        init_db()
        logger.info("Tables recreated successfully")
        return True
    except Exception as e:
        logger.error(f"Error dropping and recreating tables: {e}")
        return False

def get_user_by_id(user_id):
    """Get user by ID"""
    try:
        with db.get_cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM public.users WHERE user_id = %s",
                (user_id,)
            )
            return cur.fetchone()
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return None

def get_latest_water_data(device_id=None, limit=100):
    """Get latest water quality data with optional device filter"""
    try:
        with db.get_cursor(cursor_factory=RealDictCursor) as cur:
            if device_id:
                cur.execute("""
                    SELECT * FROM iot_water_data 
                    WHERE device_id = %s 
                    ORDER BY time DESC 
                    LIMIT %s
                """, (device_id, limit))
            else:
                cur.execute("""
                    SELECT * FROM iot_water_data 
                    ORDER BY time DESC 
                    LIMIT %s
                """, (limit,))
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error fetching water data: {e}")
        return []

def insert_water_data(device_id, ph_value, turbidity, temperature, tds_value, latitude=None, longitude=None, status='active'):
    """Insert water quality data from IoT device"""
    try:
        with db.get_cursor() as cur:
            cur.execute("""
                INSERT INTO iot_water_data 
                (device_id, ph_value, turbidity, temperature, tds_value, latitude, longitude, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING time
            """, (device_id, ph_value, turbidity, temperature, tds_value, latitude, longitude, status))
            result = cur.fetchone()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Error inserting water data: {e}")
        return None

# Test the connection when module is imported
if __name__ == "__main__":
    if db.test_connection():
        print("Database connection successful!")
        
        # Ask user if they want to reset the database
        response = input("Do you want to reset the database? (y/N): ").strip().lower()
        if response == 'y':
            if drop_and_recreate_tables():
                print("Database reset successfully!")
            else:
                print("Database reset failed!")
        else:
            # Try normal initialization
            try:
                init_db()
                print("Database initialized successfully!")
            except Exception as e:
                print(f"Normal initialization failed: {e}")
                print("Trying to reset database...")
                if drop_and_recreate_tables():
                    print("Database reset successfully!")
                else:
                    print("Database reset also failed!")
        
        # Test inserting some sample data
        try:
            sample_time = insert_water_data(
                device_id="iot_device_001",
                ph_value=7.2,
                turbidity=5.1,
                temperature=25.5,
                tds_value=120.0,
                latitude=28.6139,
                longitude=77.2090
            )
            
            if sample_time:
                print(f"Sample data inserted at: {sample_time}")
            
            # Test retrieving data
            data = get_latest_water_data(limit=5)
            print(f"Retrieved {len(data)} records")
            
        except Exception as e:
            print(f"Error testing data operations: {e}")
        
    else:
        print("Database connection failed!")