"""
Training Script - Generate face embeddings from dataset
Uses ArcFace model (via DeepFace) with GPU acceleration
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utilities
from config import MODEL_NAME, DATASET_DIR
from utils.device_manager import get_device_manager
from utils.embedding_utils import get_embedding_manager

def validate_dataset(dataset_path: str) -> bool:
    """Validate dataset structure"""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False
    
    person_dirs = [d for d in os.listdir(dataset_path)
                  if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not person_dirs:
        logger.error("No person directories found in dataset")
        return False
    
    logger.info(f"Found {len(person_dirs)} person directories")
    
    total_images = 0
    for person_name in person_dirs:
        person_path = os.path.join(dataset_path, person_name)
        images = [f for f in os.listdir(person_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not images:
            logger.warning(f"  ⚠ {person_name}: No images found")
        else:
            logger.info(f"  ✓ {person_name}: {len(images)} image(s)")
            total_images += len(images)
    
    if total_images == 0:
        logger.error("No images found in dataset")
        return False
    
    logger.info(f"Total images: {total_images}")
    return True


def train_embeddings(dataset_path: str = DATASET_DIR) -> bool:
    """
    Train face embeddings from dataset
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        True if training successful, False otherwise
    """
    try:
        logger.info("="*60)
        logger.info("FACE EMBEDDING GENERATION - START")
        logger.info("="*60)
        
        # Get device info
        device_mgr = get_device_manager()
        device_mgr.print_device_info()
        
        # Validate dataset
        if not validate_dataset(dataset_path):
            logger.error("Dataset validation failed")
            return False
        
        # Get embedding manager
        embedding_mgr = get_embedding_manager()
        
        logger.info(f"\nGenerating embeddings using {embedding_mgr.model_name}...")
        logger.info(f"Model: {embedding_mgr.model_name}")
        logger.info(f"Embedding dimension: {embedding_mgr.embedding_dim}")
        
        # Generate embeddings
        embeddings = embedding_mgr.generate_embeddings_batch(dataset_path)
        
        if not embeddings:
            logger.error("Failed to generate embeddings")
            return False
        
        # Save embeddings
        if embedding_mgr.save_embeddings():
            logger.info(f"\n✓ Training complete!")
            logger.info(f"  Persons registered: {len(embeddings)}")
            logger.info(f"  Embeddings saved to: {embedding_mgr.embeddings_db}")
            logger.info("="*60)
            return True
        else:
            logger.error("Failed to save embeddings")
            return False
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train face embeddings for recognition system"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=DATASET_DIR,
        help=f'Path to dataset directory (default: {DATASET_DIR})'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'gpu', 'cpu'],
        default='auto',
        help='Device to use for training (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Override device setting if specified
    if args.device != 'auto':
        logger.info(f"Using device: {args.device}")
    
    # Run training
    success = train_embeddings(args.dataset)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()