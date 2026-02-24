import os
from dotenv import load_dotenv

load_dotenv()

# ==================== MODEL CONFIGURATION ====================
MODEL_NAME = "ArcFace"  # Using ArcFace via DeepFace
DISTANCE_METRIC = "cosine"
THRESHOLD = 0.65  # Confidence threshold for face recognition - INCREASED for stricter matching
DETECTOR_BACKEND = "opencv"  # Face detection backend (opencv is most reliable)

# ==================== DEVICE CONFIGURATION ====================
USE_GPU = True  # Set to True to enable GPU if available
DEVICE_TYPE = "auto"  # "auto", "gpu", or "cpu"

# ==================== PATH CONFIGURATION ====================
DATABASE_DIR = "database"
DATASET_DIR = "dataset"
MODELS_DIR = "models"
LOGS_DIR = "logs"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

EMBEDDINGS_PATH = os.path.join(DATABASE_DIR, "embeddings.pkl")
ATTENDANCE_PATH = os.path.join(DATABASE_DIR, "attendance.csv")
ATTENDANCE_DB_PATH = os.path.join(DATABASE_DIR, "attendance.db")

# ==================== TRAINING CONFIGURATION ====================
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# ==================== INFERENCE CONFIGURATION ====================
INPUT_SIZE = 640  # Image resolution: 640x640 px
EMBEDDING_DIM = 512  # ArcFace embedding dimension
FRAME_SKIP = 2  # Process every nth frame
TARGET_FPS = 25  # Target inference frames per second

# ==================== LOGGING CONFIGURATION ====================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create necessary directories
for dir_path in [DATABASE_DIR, DATASET_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)