from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class XEncoderConfig:
    name: str = "uni2"  # uni2, conch, vit_base
    model_name: str = "hf-hub:MahmoodLab/UNI2-h"
    embed_dim: int = 1536
    image_size: int = 224
    freeze: bool = True


@dataclass
class PredictorConfig:
    num_layers: int = 8
    hidden_dim: int = 1024
    num_heads: int = 16
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_cls_token: bool = True
    init_from_llm: bool = False
    llm_name: str = "meta-llama/Llama-3.2-1B"


@dataclass
class YEncoderConfig:
    model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    embed_dim: int = 768
    max_length: int = 512
    freeze_base: bool = False
    lr_multiplier: float = 0.05


@dataclass
class YDecoderConfig:
    model_name: str = "gpt2"
    max_length: int = 512
    num_soft_prompt_tokens: int = 8
    freeze: bool = True  # frozen during stage 1, unfrozen during stage 2


@dataclass
class ModelConfig:
    shared_embed_dim: int = 768
    x_encoder: XEncoderConfig = field(default_factory=XEncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    y_encoder: YEncoderConfig = field(default_factory=YEncoderConfig)
    y_decoder: YDecoderConfig = field(default_factory=YDecoderConfig)


@dataclass
class DataConfig:
    data_dir: str = "data/histgen"
    image_dir: str = "data/histgen/images"
    reports_file: str = "data/histgen/reports.csv"
    split_file: str = "data/histgen/splits.json"
    image_size: int = 224
    max_report_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainConfig:
    stage: str = "pretrain"  # "pretrain" or "finetune"

    # optimization
    batch_size: int = 32
    accumulate_grad_batches: int = 1
    max_epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.01
    grad_clip_norm: float = 1.0
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])

    # infonce
    temperature: float = 0.07
    learnable_temperature: bool = True

    # precision
    precision: str = "bf16-mixed"

    # logging
    project_name: str = "histo-vl"
    run_name: str = ""
    log_every_n_steps: int = 10
    val_check_interval: float = 1.0

    # checkpointing
    checkpoint_dir: str = "checkpoints"
    save_top_k: int = 3
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"

    # hardware
    devices: int = 1
    accelerator: str = "auto"
    strategy: str = "auto"

    # resume
    resume_from: str = ""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 42


def load_config(config_path: str | Path | None = None,
                overrides: list[str] | None = None) -> Config:
    """Load config from YAML file with optional CLI overrides."""
    base = OmegaConf.structured(Config)

    if config_path is not None:
        file_conf = OmegaConf.load(config_path)
        base = OmegaConf.merge(base, file_conf)

    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        base = OmegaConf.merge(base, cli_conf)

    return OmegaConf.to_object(base)
