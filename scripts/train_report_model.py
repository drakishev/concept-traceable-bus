"""the report model training entrypoint for breast ultrasound data.

Two-stage training:
  Stage 1 (pretrain): InfoNCE alignment - image patches predict text embeddings
  Stage 2 (finetune): Decoder fine-tuning - generate reports from predicted embeddings

Usage (Stage 1, single GPU):
    python scripts/train_report_model.py --config configs/train/pretrain_report_model.yaml

Usage (Stage 1, multi-GPU):
    python scripts/train_report_model.py --config configs/train/pretrain_report_model.yaml \
        train.batch_size=16

Usage (Stage 2, load Stage 1 checkpoint):
    python scripts/train_report_model.py --config configs/train/resize_only_stage2.yaml \
        train.pretrain_checkpoint=checkpoints/report_model_stage1/best.ckpt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides")
    args = parser.parse_args()

    # ── Load and merge configs ─────────────────────────────────────────────────
    cfg = OmegaConf.load(args.config)
    if cfg.get("defaults"):
        for default in cfg.defaults:
            for key, val in default.items():
                sub_cfg_path = Path(f"configs/{key}/{val}.yaml")
                if sub_cfg_path.exists():
                    sub_cfg = OmegaConf.load(sub_cfg_path)
                    cfg = OmegaConf.merge(sub_cfg, cfg)
    if args.overrides:
        overrides = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, overrides)

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    train_cfg = cfg.train
    model_cfg = cfg.model
    data_cfg = cfg.data

    # reproducibility / multi-seed support
    seed = train_cfg.get("seed", None)
    if seed is not None:
        import lightning as _L
        _L.seed_everything(int(seed), workers=True)
        logger.info("Seed set to %d", int(seed))

    # ── Datasets ───────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader

    from src.data.datasets.bus_cot_reports import BUSCoTReportDataset, preprocess_mode_of

    root_dir = Path(".")
    preprocess_mode = preprocess_mode_of(cfg)
    logger.info("Preprocess mode: %s", preprocess_mode)

    train_dataset = BUSCoTReportDataset(
        jsonl_path=data_cfg.train_jsonl,
        root_dir=root_dir,
        image_size=data_cfg.get("image_size", 224),
        preprocess_mode=preprocess_mode,
        train=True,
    )
    logger.info("Train samples: %d", len(train_dataset))

    val_dataset = None
    val_path = data_cfg.get("val_jsonl", "")
    if val_path and Path(val_path).exists():
        val_dataset = BUSCoTReportDataset(
            jsonl_path=val_path,
            root_dir=root_dir,
            image_size=data_cfg.get("image_size", 224),
            preprocess_mode=preprocess_mode,
            train=False,
        )
        logger.info("Val samples: %d", len(val_dataset))

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    # ── Model ──────────────────────────────────────────────────────────────────
    from src.model.report_model import ConceptReportModel

    num_epochs = train_cfg.get("num_epochs", 20)
    # Estimate total steps for scheduler
    steps_per_epoch = len(train_loader)
    max_steps = train_cfg.get("max_steps", -1)
    if max_steps < 0:
        max_steps = steps_per_epoch * num_epochs

    model = ConceptReportModel(
        shared_embed_dim=model_cfg.shared_embed_dim,
        x_encoder_name=model_cfg.x_encoder_name,
        x_encoder_embed_dim=model_cfg.x_encoder_embed_dim,
        x_encoder_image_size=model_cfg.x_encoder_image_size,
        x_encoder_freeze=model_cfg.x_encoder_freeze,
        x_encoder_unfreeze_last_n=model_cfg.get('x_encoder_unfreeze_last_n', 0),
        predictor_num_layers=model_cfg.predictor_num_layers,
        predictor_hidden_dim=model_cfg.predictor_hidden_dim,
        predictor_num_heads=model_cfg.predictor_num_heads,
        predictor_mlp_ratio=model_cfg.predictor_mlp_ratio,
        predictor_dropout=model_cfg.predictor_dropout,
        predictor_use_cls_token=model_cfg.predictor_use_cls_token,
        predictor_num_query_tokens=model_cfg.get("predictor_num_query_tokens", 1),
        y_encoder_name=model_cfg.y_encoder_name,
        y_encoder_embed_dim=model_cfg.y_encoder_embed_dim,
        y_encoder_max_length=model_cfg.y_encoder_max_length,
        y_encoder_freeze_base=model_cfg.y_encoder_freeze_base,
        y_encoder_lr_multiplier=model_cfg.y_encoder_lr_multiplier,
        y_decoder_name=model_cfg.y_decoder_name,
        y_decoder_max_length=model_cfg.y_decoder_max_length,
        y_decoder_num_soft_prompt_tokens=model_cfg.y_decoder_num_soft_prompt_tokens,
        y_decoder_lora_rank=model_cfg.get("y_decoder_lora_rank", 16),
        y_decoder_lora_alpha=model_cfg.get("y_decoder_lora_alpha", 32),
        y_decoder_lora_dropout=model_cfg.get("y_decoder_lora_dropout", 0.05),
        y_decoder_force_lora=model_cfg.get("y_decoder_force_lora", False),
        stage=train_cfg.stage,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=max_steps,
        min_lr_ratio=train_cfg.get("min_lr_ratio", 0.01),
        betas=list(train_cfg.betas),
        temperature=train_cfg.get("temperature", 0.07),
        learnable_temperature=train_cfg.get("learnable_temperature", True),
        aux_heads=list(train_cfg.get("aux_heads", []) or []),
        aux_loss_weight=train_cfg.get("aux_loss_weight", 0.1),
        concept_bottleneck=train_cfg.get("concept_bottleneck", False),
        concept_dim=train_cfg.get("concept_dim", 64),
        concept_hard=train_cfg.get("concept_hard", False),
        concept_residual_dim=train_cfg.get("concept_residual_dim", 0),
    )

    # Load Stage 1 checkpoint for Stage 2
    pretrain_ckpt = train_cfg.get("pretrain_checkpoint", None)
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        logger.info("Loading Stage 1 checkpoint: %s", pretrain_ckpt)
        import torch
        state = torch.load(pretrain_ckpt, map_location="cpu")
        # Lightning checkpoint stores model state under "state_dict"
        sd = state.get("state_dict", state)
        # Drop keys whose shape differs from the current model - strict=False
        # ignores missing/unexpected keys but still RAISES on shape mismatch.
        # This lets us reuse a Stage-1 checkpoint while swapping the decoder /
        # encoder (whose weights come pretrained from HF anyway).
        own = model.state_dict()
        filtered, skipped = {}, []
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        if skipped:
            logger.warning("Skipped %d checkpoint keys (shape mismatch / absent), "
                           "e.g. %s", len(skipped), skipped[:3])
        model.load_state_dict(filtered, strict=False)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="{epoch:02d}-{val/loss:.4f}",
        monitor=train_cfg.get("monitor", "val/loss"),
        mode=train_cfg.get("monitor_mode", "min"),
        save_top_k=train_cfg.get("save_top_k", 3),
        # Every checkpoint serializes the whole model (frozen encoder + text tower
        # + full decoder), so a 32B-decoder checkpoint is ~130 GB. `last.ckpt` and
        # `final.ckpt` duplicate the monitored one and nothing reads them
        # (runner.best_ckpt resolves `epoch=*-val/loss=*.ckpt`, falling back
        # to final.ckpt only when no monitored checkpoint exists), so both are off
        # by default. Set train.save_last / train.save_final to restore them.
        save_last=train_cfg.get("save_last", False),
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_cb, lr_monitor]
    # The LR schedule is fixed by num_epochs, not by when training stops, so the
    # checkpoint kept is the one the full run would keep unless the monitored
    # metric improves again after `patience` flat epochs.
    patience = train_cfg.get("early_stop_patience")
    if patience:
        callbacks.append(EarlyStopping(monitor=checkpoint_cb.monitor, mode=checkpoint_cb.mode,
                                       patience=int(patience), verbose=True))

    # ── Logger ─────────────────────────────────────────────────────────────────
    wandb_project = train_cfg.get("wandb_project", None)
    pl_logger: list = []
    if wandb_project:
        from lightning.pytorch.loggers import WandbLogger
        pl_logger.append(WandbLogger(
            project=wandb_project,
            name=train_cfg.get("run_name", "report_model"),
        ))

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=num_epochs,
        max_steps=train_cfg.get("max_steps", -1),
        precision=train_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        val_check_interval=train_cfg.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=pl_logger if pl_logger else True,
        enable_progress_bar=True,
    )

    logger.info("Starting Stage 1 (%s) training...", train_cfg.stage)
    trainer.fit(model, train_loader, val_loader)

    # Final (last-epoch) checkpoint: redundant with the monitored one unless the
    # run saves no monitored checkpoint at all (max_steps smoke tests).
    if train_cfg.get("save_final", False) or not checkpoint_cb.best_model_path:
        final_path = output_dir / "final.ckpt"
        trainer.save_checkpoint(str(final_path))
        logger.info("Saved final checkpoint to %s", final_path)
    else:
        logger.info("Best checkpoint: %s (final.ckpt not written; set train.save_final=true "
                    "to keep a last-epoch copy)", checkpoint_cb.best_model_path)


if __name__ == "__main__":
    main()
