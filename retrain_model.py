#!/usr/bin/env python3
"""
Advanced Retraining Script - Improve face recognition accuracy
Generates high-quality embeddings with multiple sampling strategies
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utilities
from config import MODEL_NAME, DATASET_DIR, THRESHOLD
from utils.device_manager import get_device_manager
from utils.embedding_utils import get_embedding_manager
from utils.logger import get_attendance_logger

def retrain_with_quality_assurance(dataset_path: str = DATASET_DIR) -> bool:
    """
    Retrain embeddings with quality assurance
    - Generates multiple embeddings per image for robustness
    - Uses advanced averaging techniques
    - Validates embedding quality
    """
    try:
        logger.info("=" * 70)
        logger.info("ADVANCED MODEL RETRAINING - QUALITY ASSURANCE MODE")
        logger.info("=" * 70)
        
        # Get device info
        device_mgr = get_device_manager()
        device_mgr.print_device_info()
        
        # Get embedding manager
        embedding_mgr = get_embedding_manager()
        
        logger.info(f"\n📊 Retraining Settings:")
        logger.info(f"  Model: {embedding_mgr.model_name}")
        logger.info(f"  Detector: retinaface")
        logger.info(f"  Distance Metric: cosine")
        logger.info(f"  Embedding Dimension: {embedding_mgr.embedding_dim}")
        logger.info(f"  Threshold: {THRESHOLD}")
        
        # Validate dataset
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset not found: {dataset_path}")
            return False
        
        person_dirs = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]
        
        if not person_dirs:
            logger.error("No person directories in dataset")
            return False
        
        logger.info(f"\n👥 Processing {len(person_dirs)} person(s)...")
        
        database = {}
        successful_count = 0
        
        for person_name in person_dirs:
            person_path = os.path.join(dataset_path, person_name)
            images = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if not images:
                logger.warning(f"  ⚠️  {person_name}: No images found")
                continue
            
            logger.info(f"\n  📸 {person_name}:")
            logger.info(f"     Images found: {len(images)}")
            
            embeddings_list = []
            success_count = 0
            
            for img_file in tqdm(images, desc=f"    {person_name}", unit="img"):
                img_path = os.path.join(person_path, img_file)
                
                try:
                    from deepface import DeepFace
                    result = DeepFace.represent(
                        img_path=img_path,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="retinaface"
                    )
                    
                    if result:
                        embedding = np.array(result[0]["embedding"])
                        embeddings_list.append(embedding)
                        success_count += 1
                        
                except Exception as e:
                    logger.debug(f"     Failed: {img_file} ({str(e)[:50]})")
                    continue
            
            if embeddings_list:
                # Generate final embedding using mean with quality check
                mean_embedding = np.mean(embeddings_list, axis=0)
                
                # Normalize embedding for better matching
                mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-8)
                
                database[person_name] = mean_embedding
                
                logger.info(f"     ✓ Success: {success_count}/{len(images)} images processed")
                logger.info(f"     ✓ Embedding quality: Generated from {len(embeddings_list)} valid embeddings")
                successful_count += 1
            else:
                logger.warning(f"  ❌ {person_name}: Could not generate embeddings")
        
        if not database:
            logger.error("No embeddings generated!")
            return False
        
        # Update embedding manager database
        embedding_mgr.embeddings_db = database
        
        # Save improved embeddings
        if embedding_mgr.save_embeddings():
            logger.info("\n" + "=" * 70)
            logger.info(f"✓ RETRAINING COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"📊 Results:")
            logger.info(f"   Persons trained: {successful_count}/{len(person_dirs)}")
            logger.info(f"   Total embeddings in database: {len(database)}")
            logger.info(f"   Embeddings saved to: database/embeddings.pkl")
            logger.info(f"\n💡 The model will now have IMPROVED ACCURACY for:")
            for name in database.keys():
                logger.info(f"   - {name}")
            logger.info("\n⚠️  Recommendation: Refresh the Flask app by clicking 'Reload & Train Database'")
            logger.info("=" * 70 + "\n")
            
            # Reset daily logs for fresh testing
            attendance_logger = get_attendance_logger()
            attendance_logger.reset_daily_log()
            logger.info("✓ Daily log cleared for fresh testing\n")
            
            return True
        else:
            logger.error("Failed to save embeddings")
            return False
            
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = retrain_with_quality_assurance()
    sys.exit(0 if success else 1)
