"""
Device Manager - Handles GPU/CPU device selection
Automatically detects available hardware and configures TensorFlow/PyTorch accordingly
"""

import tensorflow as tf
import logging
from config import USE_GPU, DEVICE_TYPE

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages GPU/CPU device selection for deep learning inference"""
    
    def __init__(self):
        self.gpu_available = False
        self.device = "cpu"
        self.device_name = "CPU"
        self.gpu_memory_fraction = 0.7  # Use 70% of GPU memory
        self._initialize_device()
    
    def _initialize_device(self):
        """Initialize and configure device"""
        try:
            # Check if GPUs are available
            gpus = tf.config.list_physical_devices('GPU')
            self.gpu_available = len(gpus) > 0
            
            if self.gpu_available and (USE_GPU and DEVICE_TYPE != "cpu"):
                self._setup_gpu(gpus)
            else:
                self._setup_cpu()
                
        except Exception as e:
            logger.warning(f"Error during device initialization: {e}. Falling back to CPU.")
            self._setup_cpu()
    
    def _setup_gpu(self, gpus):
        """Configure GPU usage"""
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            self.device = "gpu"
            self.device_name = f"GPU ({len(gpus)} device(s))"
            logger.info(f"✓ GPU device initialized: {self.device_name}")
            logger.info(f"  GPU Details:")
            for i, gpu in enumerate(gpus):
                logger.info(f"    GPU {i}: {gpu}")
                
        except Exception as e:
            logger.warning(f"Failed to initialize GPU: {e}. Falling back to CPU.")
            self._setup_cpu()
    
    def _setup_cpu(self):
        """Configure CPU usage"""
        try:
            # Disable GPU
            tf.config.set_visible_devices([], 'GPU')
            self.device = "cpu"
            self.device_name = "CPU (Multi-core)"
            logger.info(f"✓ CPU device initialized: {self.device_name}")
        except Exception as e:
            logger.error(f"Error setting up CPU: {e}")
    
    def get_device_info(self) -> dict:
        """Get current device information"""
        return {
            "device_type": self.device,
            "device_name": self.device_name,
            "gpu_available": self.gpu_available,
            "gpu_memory_fraction": self.gpu_memory_fraction
        }
    
    def print_device_info(self):
        """Print device information to logger"""
        info = self.get_device_info()
        logger.info("="*50)
        logger.info("DEVICE CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Device Type: {info['device_type'].upper()}")
        logger.info(f"Device Name: {info['device_name']}")
        logger.info(f"Memory Fraction: {info['gpu_memory_fraction']}")
        logger.info("="*50)


# Initialize global device manager
device_manager = DeviceManager()


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance"""
    return device_manager


def is_gpu_available() -> bool:
    """Check if GPU is available and enabled"""
    return device_manager.gpu_available and device_manager.device == "gpu"


def get_device_type() -> str:
    """Get current device type ('gpu' or 'cpu')"""
    return device_manager.device
