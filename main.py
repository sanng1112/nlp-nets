"""
nlp-nets: Entry point for training and evaluation.

Usage:
    python main.py --common.config-file configs/demo.yaml
    python main.py --common.config-file configs/demo.yaml --common.sanity-check
"""

import argparse
import os
import sys
import yaml

from utils import logger
from utils.seed import set_seed
from utils.config_helper import parse_arguments, load_config


def main() -> None:
    """Main entry point: parse arguments → load config → train or evaluate."""
    parser = argparse.ArgumentParser(description="nlp-nets Training Entry Point")
    parser.add_argument("--common.config-file", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--common.sanity-check", action="store_true", help="Run sanity check only")
    parser.add_argument("--common.resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--common.eval-only", action="store_true", help="Run evaluation only (no training)")

    # Allow CLI overrides for key config values
    parser.add_argument("--model.name", type=str, default=None, help="Model name override")
    parser.add_argument("--dataset.name", type=str, default=None, help="Dataset name override")
    parser.add_argument("--optim.lr", type=float, default=None, help="Learning rate override")
    parser.add_argument("--train.batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--train.epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--train.device", type=str, default=None, help="Device override (cuda/cpu)")

    args = parser.parse_args()

    # Load YAML config
    opts = load_config(args.common_config_file)

    # Apply CLI overrides
    if args.model_name is not None:
        opts["model"]["name"] = args.model_name
    if args.dataset_name is not None:
        opts["dataset"]["name"] = args.dataset_name
    if args.optim_lr is not None:
        opts["optim"]["lr"] = args.optim_lr
    if args.train_batch_size is not None:
        opts["train"]["batch_size"] = args.train_batch_size
    if args.train_epochs is not None:
        opts["train"]["epochs"] = args.train_epochs
    if args.train_device is not None:
        opts["train"]["device"] = args.train_device

    opts["common"] = {
        "config_file": args.common_config_file,
        "sanity_check": args.common_sanity_check,
        "resume": args.common_resume,
        "eval_only": args.common_eval_only,
    }

    # Set seed for reproducibility
    set_seed(opts.get("seed", 42))

    # Log configuration
    logger.log("Configuration loaded:")
    logger.log(yaml.dump(opts, default_flow_style=False))

    # Determine task type
    task_type = opts.get("task", {}).get("type", "mlm")
    logger.log(f"Task type: {task_type}")

    # Build model
    from models.builder import build_model
    model = build_model(opts)
    logger.log(f"Model built: {opts['model']['name']}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    # Build tokenizer
    from tokenizer_factory.builder import build_tokenizer
    tokenizer = build_tokenizer(opts)

    # Build dataloaders
    from engine.data_factory import build_dataloaders
    train_loader, val_loader = build_dataloaders(opts, tokenizer)
    logger.log(f"DataLoaders built: train={len(train_loader)} batches, val={len(val_loader)} batches")

    # Build loss function
    from loss_fn import build_loss_fn
    criterion = build_loss_fn(opts)

    # Build optimizer and scheduler
    from optim import build_optimizer, build_scheduler
    optimizer = build_optimizer(opts, model)
    scheduler = build_scheduler(opts, optimizer)

    # Build metrics
    from engine.metrics_modules.builder import build_metrics
    metrics = build_metrics(task_type)

    # Sanity check mode
    if opts["common"]["sanity_check"]:
        from engine.sanity_check import run_sanity_check
        from engine.trainer import Trainer
        trainer = Trainer(
            opts=opts,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            metrics=metrics,
        )
        success = run_sanity_check(trainer)
        sys.exit(0 if success else 1)

    # Build trainer
    from engine.trainer import Trainer
    trainer = Trainer(
        opts=opts,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        metrics=metrics,
    )

    # Resume from checkpoint
    if opts["common"]["resume"]:
        trainer.load_checkpoint(opts["common"]["resume"])
        logger.log(f"Resumed from checkpoint: {opts['common']['resume']}")

    if opts["common"]["eval_only"]:
        val_loss, val_metrics = trainer.evaluate()
        logger.log(f"Evaluation complete: loss={val_loss:.4f}, metrics={val_metrics}")
        return

    # Train
    trainer.train()

    logger.log("Training complete!")


if __name__ == "__main__":
    main()
