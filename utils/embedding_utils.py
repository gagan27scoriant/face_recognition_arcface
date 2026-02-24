"""
Embedding Utilities - Handle face embedding generation and storage
"""

import os
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from deepface import DeepFace
from config import MODEL_NAME, EMBEDDINGS_PATH, DATASET_DIR, EMBEDDING_DIM, DETECTOR_BACKEND
from utils.device_manager import get_device_type
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages face embeddings - generation, storage, and loading"""
    
    def __init__(self):
        self.embeddings_db: Dict[str, np.ndarray] = {}
        self.embedding_dim = EMBEDDING_DIM
        self.model_name = MODEL_NAME
        self.load_embeddings()
    
    def load_embeddings(self) -> bool:
        """Load embeddings from pickle file"""
        try:
            if os.path.exists(EMBEDDINGS_PATH):
                with open(EMBEDDINGS_PATH, "rb") as f:
                    self.embeddings_db = pickle.load(f)
                logger.info(f"✓ Loaded {len(self.embeddings_db)} face embeddings from database")
                return True
            else:
                logger.warning(f"Embeddings file not found at {EMBEDDINGS_PATH}")
                return False
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return False
    
    def save_embeddings(self) -> bool:
        """Save embeddings to pickle file"""
        try:
            os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
            with open(EMBEDDINGS_PATH, "wb") as f:
                pickle.dump(self.embeddings_db, f)
            logger.info(f"✓ Saved {len(self.embeddings_db)} face embeddings")
            return True
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            return False
    
    def generate_embedding(self, image_path: str) -> Tuple[bool, np.ndarray]:
        """
        Generate embedding for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (success, embedding) or (False, None)
        """
        try:
            result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=False,  # Allow processing without strict face detection
                detector_backend=DETECTOR_BACKEND
            )
            if result:
                embedding = np.array(result[0]["embedding"])
                return True, embedding
            return False, None
        except Exception as e:
            logger.warning(f"Failed to generate embedding for {image_path}: {e}")
            return False, None
    
    def generate_embeddings_batch(self, dataset_path: str) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all images in dataset
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dictionary of {person_name: averaged_embedding}
        """
        database = {}
        
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path not found: {dataset_path}")
            return database
        
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        logger.info(f"Found {len(person_dirs)} persons in dataset")
        
        for person_name in person_dirs:
            person_folder = os.path.join(dataset_path, person_name)
            img_files = [f for f in os.listdir(person_folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not img_files:
                logger.warning(f"No images found for person: {person_name}")
                continue
            
            embeddings_list = []
            logger.info(f"Processing {person_name} ({len(img_files)} images)...")
            
            for img_file in tqdm(img_files, desc=person_name):
                img_path = os.path.join(person_folder, img_file)
                success, embedding = self.generate_embedding(img_path)
                
                if success:
                    embeddings_list.append(embedding)
            
            if embeddings_list:
                # Use mean embedding for better robustness
                mean_embedding = np.mean(embeddings_list, axis=0)
                database[person_name] = mean_embedding
                logger.info(f"✓ Generated embedding for {person_name} "
                           f"(avg of {len(embeddings_list)} images)")
            else:
                logger.warning(f"Could not generate any embeddings for {person_name}")
        
        self.embeddings_db = database
        return database
    
    def add_person(self, person_name: str, image_path: str) -> bool:
        """Add a single person's embedding to database"""
        try:
            success, embedding = self.generate_embedding(image_path)
            if success:
                self.embeddings_db[person_name] = embedding
                logger.info(f"✓ Added embedding for {person_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error adding person {person_name}: {e}")
            return False
    
    def remove_person(self, person_name: str) -> bool:
        """Remove a person from the embedding database"""
        try:
            if person_name in self.embeddings_db:
                del self.embeddings_db[person_name]
                logger.info(f"✓ Removed {person_name} from database")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing person {person_name}: {e}")
            return False
    
    def get_all_persons(self) -> List[str]:
        """Get list of all registered persons"""
        return list(self.embeddings_db.keys())
    
    def get_embedding(self, person_name: str) -> np.ndarray:
        """Get embedding for a specific person"""
        return self.embeddings_db.get(person_name, None)
    
    def get_database_size(self) -> int:
        """Get number of persons in database"""
        return len(self.embeddings_db)


# Global embedding manager instance
embedding_manager = EmbeddingManager()


def get_embedding_manager() -> EmbeddingManager:
    """Get the global embedding manager instance"""
    return embedding_manager
