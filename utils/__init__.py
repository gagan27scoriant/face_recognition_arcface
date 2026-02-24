"""
Utils Package - Face recognition utilities
"""

from utils.device_manager import DeviceManager, get_device_manager, is_gpu_available
from utils.embedding_utils import EmbeddingManager, get_embedding_manager
from utils.recognition import FaceRecognizer, get_face_recognizer
from utils.logger import AttendanceLogger, get_attendance_logger, log_attendance

__all__ = [
    'DeviceManager',
    'get_device_manager',
    'is_gpu_available',
    'EmbeddingManager',
    'get_embedding_manager',
    'FaceRecognizer',
    'get_face_recognizer',
    'AttendanceLogger',
    'get_attendance_logger',
    'log_attendance'
]
