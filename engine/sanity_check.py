"""
Sanity Check — dry-run before training to verify OOM, shapes, and gradient flow.
"""

import torch


def run_sanity_check(trainer) -> bool:
    """
    Run a pre-training sanity check.

    Tests:
    1. Parameter count
    2. DataLoader shape check
    3. Forward pass (OOM check)
    4. Backward pass (gradient flow)
    5. NaN checks

    Args:
        trainer: Trainer instance (must have model, criterion, optimizer, loaders).

    Returns:
        True if all checks pass, False otherwise.
    """
    print("\n" + "=" * 55)
    print("RUNNING PRE-TRAINING SANITY CHECK")
    print("=" * 55)

    # 1. Parameter Configuration Check
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} (Trainable: {trainable_params:,})")
    print(f"Device: {trainer.device} | AMP: {trainer.use_amp} | Accumulation: {trainer.accumulation_steps}")

    # 2. DataLoader & Shape Check
    print("Testing DataLoader extraction...")
    try:
        batch = next(iter(trainer.train_loader))
        inputs, targets = trainer._prepare_batch(batch)
        input_ids = inputs.get("input_ids")

        print(f"[OK] DataLoader: Input shape: {input_ids.shape}, Target present: {targets is not None}")

        # Check for NaNs
        if torch.isnan(input_ids).any():
            print("[WARNING] NaNs detected in input data!")
            return False

    except Exception as e:
        print(f"[ERROR] DataLoader failed to yield a batch. Error: {str(e)}")
        return False

    # 3. Optimizer Configuration Check
    try:
        for param_group in trainer.optimizer.param_groups:
            lr = param_group["lr"]
            opt_name = type(trainer.optimizer).__name__
            if lr > 0.1 and opt_name in ["Adam", "AdamW"]:
                print(f"[WARNING] Learning rate {lr} is extremely high for {opt_name}. Loss might diverge (NaN) immediately!")
    except Exception:
        pass

    # 4. OOM & Forward/Backward Check
    print("Simulating Forward/Backward pass (OOM & Gradient Check)...")
    try:
        trainer.model.train()
        trainer.optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=trainer.use_amp):
            outputs = trainer.model(**inputs)
            loss = outputs.get("loss")

            if loss is None:
                logits = outputs.get("logits")
                if logits is not None and targets is not None:
                    loss = trainer.criterion(None, logits, targets)

        if loss is None:
            print("[ERROR] Could not compute loss. Check model output keys.")
            return False

        if torch.isnan(loss):
            print("[WARNING] Loss is NaN in the first forward pass. Check your data or model setup!")
            return False

        print(f"[OK] Forward Pass: Initial Loss: {loss.item():.4f}")

        # Backward Pass
        if trainer.use_amp and trainer.scaler is not None:
            trainer.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient Flow Check
        has_grad = False
        for name, p in trainer.model.named_parameters():
            if p.grad is not None and torch.sum(torch.abs(p.grad)).item() > 0:
                has_grad = True
                break

        if not has_grad:
            print("[WARNING] No gradients flowing! Your model might be detached from the Loss function.")
            return False
        else:
            print("[OK] Backward Pass & Gradient Flow.")

        # Clean up
        trainer.optimizer.zero_grad()
        del inputs, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[OK] OOM Check Passed! Batch size is safe for your VRAM.")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[ERROR] OOM: Not enough VRAM for batch_size={input_ids.shape[0] if 'input_ids' in dir() else '?'}.")
            print("Suggestion: Reduce batch_size and increase gradient_accumulation_steps in config.")
            return False
        else:
            print(f"[ERROR] Forward/Backward failed. Error: {str(e)}")
            return False

    print("=" * 55)
    print("ALL SANITY CHECKS PASSED. READY TO TRAIN.")
    print("=" * 55 + "\n")
    return True
