"""
Attendance Logger - Record face recognition events to CSV/Database
"""

import pandas as pd
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from config import ATTENDANCE_PATH, ATTENDANCE_DB_PATH, DATABASE_DIR
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class AttendanceLogger:
    """Manages attendance logging to CSV and SQLite database"""
    
    def __init__(self):
        self.attendance_base_dir = DATABASE_DIR
        self.attendance_db = ATTENDANCE_DB_PATH
        self.database_dir = DATABASE_DIR
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.attendance_csv = self._get_daily_csv_path()
        
        # Deduplication tracking - STRICT MODE
        self.logged_today = {}  # Track who's been logged today: {name: True}
        self.last_logged_time = {}  # Track last log time for each person: {name: timestamp}
        self.unknown_counter = 0  # Counter for unknown faces
        self.unknown_embeddings = {}  # Store unknown face embeddings: {unique_id: embedding}
        self.unknown_similarity_threshold = 0.4  # Match threshold for unknown faces
        self.min_log_interval = 10  # STRICT: 10 seconds minimum between logging same person (was 5)
        
        self._initialize_storage()
        self._load_today_logs()
    
    def _get_daily_csv_path(self) -> str:
        """Get daily CSV file path with date in filename"""
        today = datetime.now().strftime("%Y-%m-%d")
        csv_filename = f"Attendance_{today}.csv"
        csv_path = os.path.join(self.attendance_base_dir, csv_filename)
        return csv_path
    
    def _initialize_storage(self):
        """Initialize CSV and database storage"""
        try:
            os.makedirs(self.database_dir, exist_ok=True)
            
            # Initialize CSV if it doesn't exist
            if not os.path.exists(self.attendance_csv):
                df = pd.DataFrame(columns=[
                    "Name", "Date", "Time", "Confidence", "Status"
                ])
                df.to_csv(self.attendance_csv, index=False)
                logger.info(f"✓ Created attendance CSV: {self.attendance_csv}")
            
            # Initialize SQLite database
            self._init_database()
            logger.info(f"✓ Attendance storage initialized")
            
        except Exception as e:
            logger.error(f"Error initializing storage: {e}")
    
    def _load_today_logs(self):
        """Load today's attendance from CSV to track who's been logged"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            if not os.path.exists(self.attendance_csv):
                logger.debug("No attendance CSV file found yet")
                return
            
            df = pd.read_csv(self.attendance_csv)
            today_records = df[df['Date'] == today]
            
            if today_records.empty:
                logger.debug("No records found for today")
                return
            
            # Load each person's last entry time
            for _, row in today_records.iterrows():
                name = row['Name']
                time_str = row['Time']
                
                # Mark as logged
                self.logged_today[name] = True
                
                # Try to parse time for tracking
                try:
                    entry_time = datetime.strptime(f"{today} {time_str}", "%Y-%m-%d %H:%M:%S")
                    self.last_logged_time[name] = entry_time.timestamp()
                except:
                    pass
            
            if self.logged_today:
                logger.info(f"✓ Loaded {len(self.logged_today)} people from CSV - already logged today")
        except Exception as e:
            logger.warning(f"Could not load today's logs from CSV: {e}")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        try:
            conn = sqlite3.connect(self.attendance_db)
            cursor = conn.cursor()
            
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
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    registered_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def get_next_unique_id(self) -> str:
        """Get next unique ID for unknown face"""
        self.unknown_counter += 1
        unique_id = f"Unique_{self.unknown_counter:02d}"
        logger.info(f"🔷 Assigned new unique ID: {unique_id}")
        return unique_id
    
    def find_matching_unknown_id(self, embedding: np.ndarray) -> str:
        """
        Find matching unknown ID for a face embedding using cosine similarity
        
        Args:
            embedding: Face embedding vector (512-dim for ArcFace)
            
        Returns:
            Matching Unique_ID if found, None otherwise
        """
        try:
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            # Compare with stored unknown embeddings
            for unique_id, stored_embedding in self.unknown_embeddings.items():
                if isinstance(stored_embedding, list):
                    stored_embedding = np.array(stored_embedding)
                
                if stored_embedding.ndim == 1:
                    stored_embedding = stored_embedding.reshape(1, -1)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(embedding, stored_embedding)[0][0]
                
                if similarity >= self.unknown_similarity_threshold:
                    logger.info(f"✓ Found matching unknown: {unique_id} (similarity: {similarity:.4f})")
                    return unique_id
            
            return None
            
        except Exception as e:
            logger.warning(f"Error finding matching unknown ID: {e}")
            return None
    
    def store_unknown_embedding(self, unique_id: str, embedding: np.ndarray):
        """
        Store embedding of an unknown face for future matching
        
        Args:
            unique_id: The unique ID assigned to this unknown face
            embedding: Face embedding vector
        """
        try:
            if isinstance(embedding, list):
                embedding = np.array(embedding)
            
            self.unknown_embeddings[unique_id] = embedding
            logger.debug(f"📌 Stored embedding for {unique_id}")
            
        except Exception as e:
            logger.warning(f"Error storing unknown embedding: {e}")
    
    def log_attendance(self, name: str, confidence: float = 0.0, embedding: np.ndarray = None):
        """
        Log attendance for a person ONLY ONCE PER DAY (STRICT MODE)
        
        Args:
            name: Person's name or "Unknown"
            confidence: Recognition confidence score (0-1)
            embedding: Face embedding vector for unknown person matching
            
        Returns:
            Tuple of (success, logged_name)
        """
        try:
            current_time = datetime.now()
            now_timestamp = current_time.timestamp()
            
            # ========== STRICT DEDUPLICATION ==========
            # Check if already logged today - IMMEDIATE RETURN if yes
            if name in self.logged_today:
                logger.debug(f"⏭️ {name} already logged today - REJECT")
                return False, name
            
            # Handle unknown faces - use smart matching if embedding available
            if name == "Unknown":
                if embedding is not None:
                    # Try to find matching unknown
                    matching_id = self.find_matching_unknown_id(embedding)
                    if matching_id:
                        # Check if this matching unknown was already logged
                        if matching_id in self.logged_today:
                            logger.debug(f"⏭️ {matching_id} (matched unknown) already logged - REJECT")
                            return False, matching_id
                        name = matching_id
                    else:
                        # Assign new unique ID and store embedding
                        name = self.get_next_unique_id()
                        self.store_unknown_embedding(name, embedding)
                else:
                    # No embedding available, assign new unique ID
                    name = self.get_next_unique_id()
            
            # ========== FINAL CHECK before logging ==========
            # Double-check person not already logged
            if name in self.logged_today:
                logger.debug(f"⏭️ {name} already logged (double-check) - REJECT")
                return False, name
            
            # Mark as logged FIRST (before any I/O)
            self.logged_today[name] = True
            self.last_logged_time[name] = now_timestamp
            
            record = {
                "Name": name,
                "Date": current_time.strftime("%Y-%m-%d"),
                "Time": current_time.strftime("%H:%M:%S"),
                "Confidence": f"{confidence:.4f}",
                "Status": "present"
            }
            
            # Log ONLY to CSV (single source of truth)
            try:
                if os.path.exists(self.attendance_csv):
                    df = pd.read_csv(self.attendance_csv)
                else:
                    df = pd.DataFrame(columns=["Name", "Date", "Time", "Confidence", "Status"])
                
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                df.to_csv(self.attendance_csv, index=False)
                logger.info(f"✅ LOGGED (ONCE): {name} ({confidence:.2f}) → {os.path.basename(self.attendance_csv)}")
                
            except Exception as csv_error:
                logger.error(f"Error writing to CSV: {csv_error}")
                # Remove from logged_today if CSV write failed
                del self.logged_today[name]
                del self.last_logged_time[name]
                return False, name
            
            return True, name
            
        except Exception as e:
            logger.error(f"Error logging attendance: {e}")
            return False, name
    
    def log_batch_attendance(self, records: List[Dict]) -> int:
        """
        Log multiple attendance records
        
        Args:
            records: List of attendance records
            
        Returns:
            Number of records logged successfully
        """
        count = 0
        for record in records:
            success, _ = self.log_attendance(
                record.get('name'),
                record.get('confidence', 0.0)
            )
            if success:
                count += 1
        return count
    
    def get_daily_attendance(self, date: str = None) -> pd.DataFrame:
        """
        Get attendance records for a specific date from CSV
        
        Args:
            date: Date in format YYYY-MM-DD (current date if None)
            
        Returns:
            DataFrame of attendance records
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            if not os.path.exists(self.attendance_csv):
                return pd.DataFrame(columns=["Name", "Date", "Time", "Confidence", "Status"])
            
            df = pd.read_csv(self.attendance_csv)
            df_filtered = df[df['Date'] == date]
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error getting daily attendance: {e}")
            return pd.DataFrame()
    
    def get_person_attendance(self, person_name: str, days: int = 7) -> pd.DataFrame:
        """
        Get attendance records for a person in last N days
        
        Args:
            person_name: Person's name
            days: Number of days to retrieve
            
        Returns:
            DataFrame of attendance records
        """
        try:
            conn = sqlite3.connect(self.attendance_db)
            query = """
                SELECT * FROM attendance 
                WHERE name = ? AND 
                datetime(timestamp) >= datetime('now', '-' || ? || ' days')
                ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, conn, params=(person_name, days))
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting person attendance: {e}")
            return pd.DataFrame()
    
    def get_summary_report(self, date: str = None) -> Dict:
        """
        Get attendance summary report from CSV
        
        Args:
            date: Date in format YYYY-MM-DD (current date if None)
            
        Returns:
            Dictionary with attendance statistics
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y-%m-%d")
            
            df = self.get_daily_attendance(date)
            
            if df.empty:
                return {
                    "date": date,
                    "total_entries": 0,
                    "unique_persons": 0,
                    "avg_confidence": 0.0,
                    "records": []
                }
            
            # Convert confidence column to numeric (in case it's stored as string)
            df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
            
            return {
                "date": date,
                "total_entries": len(df),
                "unique_persons": df['Name'].nunique(),
                "persons": df['Name'].unique().tolist(),
                "avg_confidence": float(df['Confidence'].mean()),
                "records": df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "total_entries": 0,
                "unique_persons": 0,
                "avg_confidence": 0.0,
                "records": []
            }
    
    def clear_old_records(self, days: int = 90) -> int:
        """
        Delete attendance records older than N days
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of records deleted
        """
        try:
            conn = sqlite3.connect(self.attendance_db)
            cursor = conn.cursor()
            
            cursor.execute("""
                DELETE FROM attendance 
                WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
            """, (days,))
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"✓ Deleted {deleted} old attendance records")
            return deleted
            
        except Exception as e:
            logger.error(f"Error clearing old records: {e}")
            return 0
    
    def reset_daily_log(self):
        """Reset the daily log (useful for new day)"""
        self.logged_today = {}
        self.last_logged_time = {}
        self.unknown_counter = 0
        self.unknown_embeddings = {}
        logger.info("✓ Daily log reset - ready for new day")


# Global attendance logger instance
attendance_logger = AttendanceLogger()


def log_attendance(name: str, confidence: float = 0.0) -> bool:
    """Log attendance for a person (backward compatible)"""
    success, _ = attendance_logger.log_attendance(name, confidence)
    return success


def get_attendance_logger() -> AttendanceLogger:
    """Get the global attendance logger instance"""
    return attendance_logger