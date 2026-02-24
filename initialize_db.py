"""
Initialize Database and Directories
Set up initial database schema and directory structure
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    DATABASE_DIR, DATASET_DIR, MODELS_DIR, LOGS_DIR,
    ATTENDANCE_PATH, ATTENDANCE_DB_PATH
)


def initialize_directories():
    """Create necessary directories"""
    directories = [
        DATABASE_DIR,
        DATASET_DIR,
        MODELS_DIR,
        LOGS_DIR,
        os.path.join(DATASET_DIR, 'person1'),
        os.path.join(DATASET_DIR, 'person2'),
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        if os.path.exists(dir_path):
            logger.info(f"✓ Directory created/verified: {dir_path}")
        else:
            logger.error(f"✗ Failed to create directory: {dir_path}")
            return False
    
    return True


def initialize_csv():
    """Initialize CSV files"""
    try:
        if not os.path.exists(ATTENDANCE_PATH):
            df = pd.DataFrame(columns=[
                "Name", "Date", "Time", "Confidence", "Status"
            ])
            df.to_csv(ATTENDANCE_PATH, index=False)
            logger.info(f"✓ Attendance CSV created: {ATTENDANCE_PATH}")
        else:
            logger.info(f"✓ Attendance CSV exists: {ATTENDANCE_PATH}")
        return True
    except Exception as e:
        logger.error(f"✗ Error creating CSV: {e}")
        return False


def initialize_database():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(ATTENDANCE_DB_PATH)
        cursor = conn.cursor()
        
        # Create attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                confidence REAL,
                status TEXT DEFAULT 'present',
                device TEXT
            )
        """)
        logger.info(f"✓ Attendance table created")
        
        # Create persons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                registered_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        """)
        logger.info(f"✓ Persons table created")
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON attendance(name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON attendance(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance(timestamp)")
        logger.info(f"✓ Indexes created")
        
        conn.commit()
        conn.close()
        logger.info(f"✓ Database initialized: {ATTENDANCE_DB_PATH}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error initializing database: {e}")
        return False


def main():
    """Main initialization function"""
    logger.info("="*60)
    logger.info("DATABASE AND DIRECTORY INITIALIZATION")
    logger.info("="*60)
    
    success = True
    
    # Initialize directories
    if not initialize_directories():
        success = False
    
    # Initialize CSV
    if not initialize_csv():
        success = False
    
    # Initialize database
    if not initialize_database():
        success = False
    
    logger.info("="*60)
    if success:
        logger.info("✓ Initialization complete!")
        logger.info("\nNext steps:")
        logger.info("1. Add training images to dataset/ directory")
        logger.info("2. Run: python train.py")
        logger.info("3. Run: python app.py")
        logger.info("="*60)
        return 0
    else:
        logger.error("✗ Initialization failed!")
        logger.info("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
