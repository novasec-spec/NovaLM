"""
NovaLM Configuration Module

Central configuration for model architecture, training hyperparameters,
data pipeline, and generation settings.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List
import json


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelConfig:
    """Hyperparameters for NovaLM architecture."""
    vocab_size: int = 16000
    max_sequence_length: int = 256
    embedding_dim: int = 256
    num_heads: int = 8
    feed_forward_dim: int = 1024
    num_layers: int = 6
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainingConfig:
    """Hyperparameters for continuous model training."""
    batch_size: int = 8
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_clip_norm: float = 1.0
    weight_decay: float = 0.01

    eval_interval_steps: int = 200
    eval_batches: int = 25
    checkpoint_interval_steps: int = 500
    save_top_k: int = 3

    seed: int = 42

    def to_dict(self):
        return asdict(self)


# ============================================================
# DATA PIPELINE CONFIGURATION
# ============================================================

@dataclass
class DataConfig:
    """Paths and settings for incremental data processing."""
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    tokens_dir: Path = PROJECT_ROOT / "data" / "processed" / "tokens"
    manifest_file: Path = PROJECT_ROOT / "data" / "processed" / "manifest.json"

    train_bin: Path = PROJECT_ROOT / "data" / "processed" / "tokens" / "train.bin"
    val_bin: Path = PROJECT_ROOT / "data" / "processed" / "tokens" / "val.bin"

    validation_ratio: float = 0.10
    sequence_length: int = 64
    min_characters: int = 40
    min_words: int = 10

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


# ============================================================
# GENERATION CONFIGURATION
# ============================================================

@dataclass
class GenerationConfig:
    """Settings for autoregressive text generation."""
    max_new_tokens: int = 60
    temperature: float = 0.8
    top_k: int = 40
    repetition_penalty: float = 1.1

    def to_dict(self):
        return asdict(self)


# ============================================================
# MASTER CONFIG
# ============================================================

@dataclass
class NovaLMConfig:
    """Master configuration encapsulating all subsystems."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    tokenizer_file: Path = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
    tokenizer_metadata_file: Path = PROJECT_ROOT / "tokenizer" / "tokenizer_metadata.json"

    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    latest_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "latest.pt"
    best_checkpoint: Path = PROJECT_ROOT / "checkpoints" / "best.pt"

    history_file: Path = PROJECT_ROOT / "evaluation" / "training_history.csv"
    plot_file: Path = PROJECT_ROOT / "evaluation" / "training_plots.png"

    def to_dict(self):
        return {
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict(),
            "generation": self.generation.to_dict(),
            "tokenizer_file": str(self.tokenizer_file),
            "latest_checkpoint": str(self.latest_checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "history_file": str(self.history_file),
            "plot_file": str(self.plot_file),
        }

    def save_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


# Default shared instance
default_config = NovaLMConfig()
