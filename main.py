"""
nlp-nets: Entry point for training and evaluation.

Usage:
    python main.py --common.config-file configs/demo.yaml
    python main.py --common.config-file configs/demo.yaml --common.sanity-check
"""

import sys
import yaml

from utils import logger
from utils.seed import set_seed
from utils.config_helper import parse_arguments


def main() -> None:
    """Main entry point: parse arguments → load config → train or evaluate."""
    _, opts = parse_arguments()

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
