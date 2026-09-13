"""Configuration dataclasses for TurathiAI.

The baseline configuration reflects the paper's best setup (Section 4.3):
LoRA rank 32, Prodigy optimizer, constant LR schedule, masked loss, and
min-SNR loss weighting on the SDXL backbone. Ablations override a single field.

Configs can be loaded from YAML with :func:`load_config`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    dataset_dir: str = "data/CHDB_Full"
    resolution: int = 1024
    mask_suffix: str = "-masklabel.png"
    caption_ext: str = ".txt"
    image_exts: tuple[str, ...] = ("jpg", "jpeg", "png")
    val_split: float = 0.10           # ~63 of 634 sets held out
    use_masked_loss: bool = True      # masked training (Section 3.1.3)
    random_flip: bool = True
    seed: int = 42


@dataclass
class ModelConfig:
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    lora_rank: int = 32               # baseline rank
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # UNet attention projections adapted by LoRA
    target_modules: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")


@dataclass
class TrainConfig:
    output_dir: str = "outputs/lora_default_best"
    optimizer: str = "prodigy"        # "prodigy" | "adamw8bit" | "adamw"
    learning_rate: float = 1.0        # Prodigy is learning-rate-free (uses 1.0)
    lr_scheduler: str = "constant"    # "constant" | "cosine" | "linear"
    lr_warmup_steps: int = 0
    max_train_steps: int = 9000       # ablations trained for 9,000 steps
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    # min-SNR loss weighting (Hang et al., 2023). Set snr_gamma=None to disable.
    snr_gamma: Optional[float] = 5.0
    checkpointing_steps: int = 500
    validation_steps: int = 250
    save_best_on_val: bool = True
    seed: int = 42


@dataclass
class EvalConfig:
    test_prompts: str = "data/test_prompts.txt"
    eval_steps: int = 25
    guidance_scale: float = 6.0
    images_per_prompt: int = 4        # paper protocol: 100 prompts x 4 seeds = 400
    resolution: int = 1024
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    fid_feature: int = 2048
    lpips_net: str = "alex"


@dataclass
class Config:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file (nested keys override defaults)."""
    import yaml

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg = Config()
    cfg.name = raw.get("name", cfg.name)
    for section, klass in (("data", DataConfig), ("model", ModelConfig),
                           ("train", TrainConfig), ("eval", EvalConfig)):
        overrides = raw.get(section, {}) or {}
        base = getattr(cfg, section)
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
    return cfg
