"""
config.py
=========
Centralized configuration for the LearnFlow LSTM system.
Defines data pipeline, model architecture, training, and API settings.
"""

from dataclasses import dataclass
from pathlib import Path
import os

# ─────────────────────────────────────────────────────────────────────────────
# DATA CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    """Data pipeline parameters."""
    seq_len: int = 10                        # Sequence length (timesteps per learner)
    n_features: int = 6                      # Numerical features (quiz, engagement, hints, duration, correct, incorrect)
    n_modules: int = 6                       # Number of course modules
    n_synthetic_learners: int = 500          # Synthetic data generation size
    test_split: float = 0.2
    val_split: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Model architecture and checkpoint parameters."""
    embed_dim: int = 8                       # Embedding dimension for categorical features
    lstm_units_1: int = 128                  # First LSTM layer units
    lstm_units_2: int = 64                   # Second LSTM layer units
    attention_units: int = 64                # Attention mechanism units
    dropout_rate: float = 0.3                # Dropout rate
    l2_reg: float = 1e-4                     # L2 regularization coefficient
    
    # Loss weights for multi-task learning
    loss_weight_performance: float = 0.3     # Performance score task weight
    loss_weight_mastery: float = 0.35        # Mastery probability task weight
    loss_weight_dropout: float = 0.35        # Dropout risk task weight
    
    checkpoint_name: str = "best_model.keras"
    scaler_name: str = "scaler.npy"
    version: str = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Training loop and optimization parameters."""
    epochs: int = 50                         # Maximum training epochs
    batch_size: int = 32                     # Mini-batch size
    learning_rate: float = 1e-3              # Adam learning rate
    clipnorm: float = 1.0                    # Gradient clipping norm
    
    # Early stopping
    es_monitor: str = "val_loss"
    es_patience: int = 10
    restore_best: bool = True
    
    # Learning rate decay
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6
    
    # Reproducibility
    seed: int = 42
    
    # MLflow logging (optional)
    mlflow_experiment: str = "lstm-learner-progression"
    log_artifacts: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class APIConfig:
    """API server and middleware parameters."""
    host: str = "localhost"
    port: int = 8000
    workers: int = 1
    log_level: str = "info"
    api_key: str = os.getenv("LEARNFLOW_API_KEY", "dev-insecure-key-change-me")
    rate_limit_rpm: int = 60


# ─────────────────────────────────────────────────────────────────────────────
# PATHS CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PathsConfig:
    """Directory and file paths."""
    root: Path = Path(__file__).parent.parent.parent
    data: Path = None
    raw: Path = None
    processed: Path = None
    sequences: Path = None
    checkpoints: Path = None
    logs: Path = None
    tensorboard: Path = None
    
    def __post_init__(self):
        """Initialize derived paths."""
        if self.data is None:
            self.data = self.root / "data"
        if self.raw is None:
            self.raw = self.data / "raw"
        if self.processed is None:
            self.processed = self.data / "processed"
        if self.sequences is None:
            self.sequences = self.data / "sequences"
        if self.checkpoints is None:
            self.checkpoints = self.root / "artifacts" / "checkpoints"
        if self.logs is None:
            self.logs = self.root / "logs"
        if self.tensorboard is None:
            self.tensorboard = self.logs / "tensorboard"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONFIG OBJECT (Singleton)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """Top-level configuration container."""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    api: APIConfig = None
    paths: PathsConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.api is None:
            self.api = APIConfig()
        if self.paths is None:
            self.paths = PathsConfig()


# Global config instance
cfg = Config()
