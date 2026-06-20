import os
import time
import math
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import logger


class Trainer:
    """
    General-purpose trainer for NLP models.

    Supports:
    - Mixed precision (AMP)
    - Gradient accumulation
    - DDP (Distributed Data Parallel)
    - Checkpointing and resuming
    - Metric tracking
    - Logging (console + optional wandb)
    """

    def __init__(
        self,
        opts: Dict[str, Any],
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        tokenizer: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self.opts = opts
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.metrics = metrics

        # Training config
        train_config = opts.get("train", {})
        self.device = torch.device(train_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.epochs = train_config.get("epochs", 3)
        self.batch_size = train_config.get("batch_size", 32)
        self.accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.use_amp = train_config.get("use_amp", True) and self.device.type == "cuda"
        self.log_interval = train_config.get("log_interval", 10)
        self.save_every = train_config.get("save_every", 1000)
        self.output_dir = train_config.get("output_dir", "checkpoints")

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Move model to device
        self.model.to(self.device)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Track best metrics
        self.best_loss = float("inf")
        self.best_metric = 0.0
        self.global_step = 0
        self.epoch = 0

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns:
            Dict with final training metrics.
        """
        logger.print_header("Starting Training")

        for epoch in range(self.epochs):
            self.epoch = epoch
            train_metrics = self._train_epoch()

            # Validation
            val_loss, val_metrics = self.evaluate()

            # Log
            log_msg = (
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )
            if val_metrics:
                log_msg += f" | Val Metrics: {val_metrics}"

            logger.log(log_msg)

            # Save checkpoint if best
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint("best.pt")

        logger.print_header("Training Complete")
        return {"train_loss": train_metrics.get("loss", 0), "val_loss": self.best_loss}

    def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader) if self.train_loader else 0

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {self.epoch + 1}/{self.epochs}",
            leave=False,
        )

        for batch_idx, batch in progress_bar:
            # Prepare batch
            inputs, targets = self._prepare_batch(batch)

            # Forward pass with AMP
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(**inputs)
                loss = outputs.get("loss")

                if loss is None:
                    # Fallback: compute loss manually
                    logits = outputs.get("logits")
                    if logits is not None and targets is not None:
                        loss = self.criterion(None, logits, targets)

                if loss is None:
                    raise ValueError("Model output must contain 'loss' or 'logits'")

                # Normalize loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation step
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Step scheduler
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        pass  # Call after validation
                    else:
                        self.scheduler.step()

            # Track loss
            total_loss += loss.item() * self.accumulation_steps

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item() * self.accumulation_steps:.4f}"})

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {"loss": avg_loss}

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Optional[Dict[str, float]]]:
        """
        Evaluate the model on the validation set.

        Returns:
            Tuple of (average loss, metrics dict or None).
        """
        if self.val_loader is None:
            return 0.0, None

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        if self.metrics is not None:
            self.metrics.to(self.device)
            self.metrics.reset()

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            inputs, targets = self._prepare_batch(batch)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                outputs = self.model(**inputs)
                loss = outputs.get("loss")

                if loss is None:
                    logits = outputs.get("logits")
                    if logits is not None and targets is not None:
                        loss = self.criterion(None, logits, targets)

                if loss is not None:
                    total_loss += loss.item()

                # Update metrics
                if self.metrics is not None and targets is not None:
                    logits = outputs.get("logits")
                    if logits is not None:
                        predictions = logits.argmax(dim=-1)
                        self.metrics.update(predictions, targets)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        metric_results = None
        if self.metrics is not None:
            try:
                metric_results = self.metrics.compute()
                metric_results = {k: v.item() if torch.is_tensor(v) else v for k, v in metric_results.items()}
            except Exception:
                pass

        self.model.train()
        return avg_loss, metric_results

    def _prepare_batch(self, batch: Any) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        """
        Prepare a batch for model input and loss computation.

        Args:
            batch: Raw batch from DataLoader.

        Returns:
            Tuple of (input_dict, target_tensor_or_None).
        """
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                input_ids, labels = batch
                inputs = {"input_ids": input_ids.to(self.device)}
                targets = labels.to(self.device) if labels is not None else None
            elif len(batch) == 3:
                input_ids, attention_mask, labels = batch
                inputs = {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                }
                targets = labels.to(self.device) if labels is not None else None
            else:
                raise ValueError(f"Unexpected batch length: {len(batch)}")
        elif isinstance(batch, dict):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != "labels"}
            targets = batch.get("labels")
            if targets is not None:
                targets = targets.to(self.device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        return inputs, targets

    def _save_checkpoint(self, filename: str) -> str:
        """Save a model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.opts,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.log(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a model checkpoint."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.log(f"Checkpoint loaded: {path} (epoch {self.epoch})")
