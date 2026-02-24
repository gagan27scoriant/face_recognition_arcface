"""
Face Recognition - Real-time face detection and recognition
"""

import cv2
import numpy as np
import logging
from threading import Lock
from typing import Tuple, List
import time
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from config import MODEL_NAME, DISTANCE_METRIC, THRESHOLD, INPUT_SIZE, DETECTOR_BACKEND
from utils.embedding_utils import get_embedding_manager
from utils.device_manager import is_gpu_available, get_device_type

logger = logging.getLogger(__name__)


class FaceRecognizer:
    """Real-time face recognition engine using ArcFace embeddings"""
    
    def __init__(self):
        self.embedding_manager = get_embedding_manager()
        self.model_name = MODEL_NAME
        self.distance_metric = DISTANCE_METRIC
        self.threshold = THRESHOLD
        self.detector_backend = DETECTOR_BACKEND
        self.lock = Lock()
        self.recognition_cache = {}
        self.cache_timeout = 5  # seconds
        self.last_cache_time = {}
        
        logger.info(f"FaceRecognizer initialized with {self.embedding_manager.get_database_size()} registered faces")
        logger.info(f"Device: {get_device_type().upper()}")
        logger.info(f"Using threshold: {self.threshold}")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for better detection"""
        # Resize frame for faster processing
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_height = 480
        new_width = int(new_height * aspect_ratio)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        return frame_resized
    
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces in frame using YOLOv8
        
        Returns:
            List of detected faces with bounding boxes
        """
        try:
            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using DeepFace
            detected_faces = DeepFace.extract_faces(
                img_path=frame_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            return detected_faces
            
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            return []
    
    def recognize_face(self, face_embedding: np.ndarray) -> Tuple[str, float]:
        """
        Recognize a face from its embedding
        
        Args:
            face_embedding: 512-dimensional face embedding
            
        Returns:
            Tuple of (person_name, confidence_score)
        """
        try:
            database = self.embedding_manager.embeddings_db
            
            if not database:
                logger.warning("No registered faces in database")
                return "Unknown", 0.0
            
            best_match = "Unknown"
            best_score = -1
            
            # Compare against all registered embeddings
            for person_name, db_embedding in database.items():
                # Use cosine similarity (0 to 1, where 1 is identical)
                similarity = cosine_similarity(
                    [face_embedding],
                    [db_embedding]
                )[0][0]
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = person_name
            
            # Check if best match meets threshold
            confidence = best_score if best_score > self.threshold else 0.0
            
            if confidence > 0:
                return best_match, float(confidence)
            else:
                return "Unknown", 0.0
                
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return "Unknown", 0.0
    
    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[dict]:
        """
        Detect and recognize all faces in a frame
        
        Args:
            frame: OpenCV frame
            
        Returns:
            List of recognized faces with details
        """
        recognized_faces = []
        
        try:
            # Detect faces
            detected_faces = self.detect_faces(frame)
            
            for face_data in detected_faces:
                try:
                    face_image = face_data['face']
                    x, y, w, h = face_data['facial_area']['x'], \
                                 face_data['facial_area']['y'], \
                                 face_data['facial_area']['w'], \
                                 face_data['facial_area']['h']
                    
                    # Generate embedding for detected face
                    face_rgb = cv2.cvtColor(
                        (face_image * 255).astype(np.uint8),
                        cv2.COLOR_BGR2RGB
                    ) if len(face_image.shape) == 3 else (face_image * 255).astype(np.uint8)
                    
                    embedding = DeepFace.represent(
                        img_path=face_rgb,
                        model_name=self.model_name,
                        enforce_detection=False
                    )[0]["embedding"]
                    
                    # Recognize face
                    person_name, confidence = self.recognize_face(np.array(embedding))
                    
                    recognized_faces.append({
                        'name': person_name,
                        'confidence': confidence,
                        'embedding': embedding,
                        'x': x,
                        'y': y,
                        'w': w,
                        'h': h,
                        'timestamp': time.time()
                    })
                    
                except Exception as e:
                    logger.debug(f"Error recognizing individual face: {e}")
                    continue
            
            return recognized_faces
            
        except Exception as e:
            logger.error(f"Error in recognize_faces_in_frame: {e}")
            return []
    
    def draw_recognition_results(self, frame: np.ndarray, 
                                  recognized_faces: List[dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: OpenCV frame
            recognized_faces: List of recognized faces
            
        Returns:
            Frame with drawn annotations
        """
        annotated_frame = frame.copy()
        
        for face in recognized_faces:
            x, y, w, h = face['x'], face['y'], face['w'], face['h']
            confidence = face['confidence']
            name = face['name']
            
            # Determine color based on recognition confidence
            if name != "Unknown" and confidence > self.threshold:
                color = (0, 255, 0)  # Green for recognized
                thickness = 2
            else:
                color = (0, 0, 255)  # Red for unknown
                thickness = 2
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw label and confidence
            label = f"{name} ({confidence:.2f})" if confidence > 0 else "Unknown"
            cv2.putText(annotated_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        return annotated_frame
    
    def update_database(self):
        """Reload embeddings from database"""
        self.embedding_manager.load_embeddings()
        logger.info(f"Database updated. Now tracking {self.embedding_manager.get_database_size()} persons")


# Global face recognizer instance
face_recognizer = FaceRecognizer()


def get_face_recognizer() -> FaceRecognizer:
    """Get the global face recognizer instance"""
    return face_recognizer