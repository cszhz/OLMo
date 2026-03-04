"""Train OLMo with optimum-neuron on AWS Trainium/Inferentia.

Run this script with 'torchrun':
    torchrun --nproc_per_node=32 scripts/train-optimum-neuron.py configs/your-config.yaml
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
import importlib.util

import torch
from transformers import HfArgumentParser

from olmo.config import TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoConfigurationError
from olmo.torch_util import (
    barrier,
    get_global_rank,
    get_local_rank,
    seed_all,
)
from olmo.util import (
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

# Import OLMo neuron wrapper
sys.path.insert(0, str(Path(__file__).parent.parent))
from olmo_neuron_wrapper import OLMoForCausalLM

# Workaround for broken peft imports in optimum-neuron
# The issue: optimum.neuron has version incompatibility with peft package
# Solution: Create a complete mock of the peft module before importing optimum.neuron
import types

# Create mock peft module hierarchy with necessary stubs
mock_peft = types.ModuleType('optimum.neuron.peft')
mock_peft_tuners = types.ModuleType('optimum.neuron.peft.tuners')
mock_peft_lora = types.ModuleType('optimum.neuron.peft.tuners.lora')
mock_peft_layer = types.ModuleType('optimum.neuron.peft.tuners.lora.layer')

# Add mock classes/functions that other modules expect
class NeuronPeftModel:
    """Mock class - not used in our training"""
    pass

def get_peft_model(*args, **kwargs):
    """Mock function - not used in our training"""
    raise NotImplementedError("LoRA/PEFT not used in this training script")

# Inject mocks into peft module
mock_peft.NeuronPeftModel = NeuronPeftModel
mock_peft.get_peft_model = get_peft_model

# Register mocks in sys.modules
sys.modules['optimum.neuron.peft'] = mock_peft
sys.modules['optimum.neuron.peft.tuners'] = mock_peft_tuners
sys.modules['optimum.neuron.peft.tuners.lora'] = mock_peft_lora
sys.modules['optimum.neuron.peft.tuners.lora.layer'] = mock_peft_layer

# Now import optimum-neuron classes - should work with mocked peft
from optimum.neuron import NeuronTrainingArguments, NeuronTrainer

log = logging.getLogger("train")


def convert_olmo_to_neuron_config(cfg: TrainConfig) -> NeuronTrainingArguments:
    """Convert OLMo TrainConfig to NeuronTrainingArguments."""

    # OLMo is not a custom neuron modeling model, so we use data parallelism only
    # tensor_parallel_size must be 1 for non-optimum models
    tensor_parallel_size = 1

    # Create neuron training config with only supported parameters
    neuron_args = NeuronTrainingArguments(
        output_dir=cfg.save_folder,
        overwrite_output_dir=cfg.save_overwrite,

        # Training parameters
        max_steps=cfg.max_duration if cfg.max_duration else -1,
        per_device_train_batch_size=cfg.device_train_microbatch_size,
        gradient_accumulation_steps=cfg.device_train_grad_accum,

        # Learning rate and scheduler
        learning_rate=cfg.optimizer.learning_rate,
        warmup_steps=int(cfg.scheduler.t_warmup) if cfg.scheduler.t_warmup else 0,
        weight_decay=cfg.optimizer.weight_decay,

        # Precision
        bf16=cfg.precision == "amp_bf16",

        # Neuron-specific parameters
        tensor_parallel_size=tensor_parallel_size,
        zero_1=True,  # Enable ZeRO-1 optimizer

        # Disable gradient clipping (causes issues with wrapped models)
        max_grad_norm=0.0,  # 0.0 means no clipping

        # Checkpointing
        save_strategy="steps",
        save_steps=cfg.save_interval if cfg.save_interval else 100,

        # Logging
        logging_steps=cfg.console_log_interval if cfg.console_log_interval else 1,

        # Other settings
        do_train=True,
        seed=cfg.seed,
    )

    return neuron_args


class OLMoNeuronTrainer(NeuronTrainer):
    """Custom Neuron trainer for OLMo that uses OLMo's data loader."""

    def __init__(self, olmo_config, train_loader, *args, **kwargs):
        self.olmo_config = olmo_config
        self.olmo_train_loader = train_loader
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        """Use OLMo's custom data loader instead of HF's default."""
        return self.olmo_train_loader

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using OLMo's forward method."""
        # OLMo's data format
        batch = inputs

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
        )

        # Compute loss
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self):
        """Override checkpoint saving to avoid multi-process Neuron Runtime bug.

        The bug: Neuron Runtime crashes during collective communication when saving
        checkpoints in multi-process training with assertion error:
        'Assertion `it != bootstrap_participants.end()' failed'

        Workaround: Only save on rank 0 and disable distributed checkpoint features.
        """
        # Get current rank
        import torch.distributed as dist
        if dist.is_initialized():
            local_rank = dist.get_rank()
        else:
            local_rank = 0

        # Only save on rank 0 to avoid collective communication
        if local_rank != 0:
            # Non-primary ranks: skip saving
            return

        # For rank 0: call parent's save but with safe settings
        checkpoint_folder = f"{self.args.output_dir}/checkpoint-{self.state.global_step}"

        # Save model using our custom save method that avoids collectives
        self._save_model_safe(checkpoint_folder)

        # Save optimizer, scheduler, and trainer state (rank 0 only)
        if self.args.should_save:
            import os
            os.makedirs(checkpoint_folder, exist_ok=True)

            # Save trainer state
            self.state.save_to_json(os.path.join(checkpoint_folder, "trainer_state.json"))

            # Save optimizer state
            optimizer_path = os.path.join(checkpoint_folder, "optimizer.pt")
            torch.save(self.optimizer.state_dict(), optimizer_path)

            # Save scheduler state
            scheduler_path = os.path.join(checkpoint_folder, "scheduler.pt")
            torch.save(self.lr_scheduler.state_dict(), scheduler_path)

            # Save training args
            torch.save(self.args, os.path.join(checkpoint_folder, "training_args.bin"))

            print(f"Checkpoint saved to {checkpoint_folder}")

    def _save_model_safe(self, output_dir):
        """Safely save model on rank 0 only, avoiding collective communication."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Get the underlying model (unwrap if needed)
        model_to_save = self.model
        if hasattr(model_to_save, 'module'):
            model_to_save = model_to_save.module

        # Use our wrapper's save_pretrained which is safe for single process
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(
                output_dir,
                is_main_process=True,
                state_dict=model_to_save.state_dict() if hasattr(model_to_save, 'state_dict') else None,
                save_function=torch.save,
            )
        else:
            # Fallback: save state dict directly
            torch.save(
                model_to_save.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin")
            )

    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to only save on rank 0."""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            # Non-primary ranks: skip
            return

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self._save_model_safe(output_dir)


def main(cfg: TrainConfig) -> None:
    """Main training function using optimum-neuron."""

    # Ensure run name set
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    barrier()

    # Set environment variables for Neuron
    os.environ["NEURON_CC_FLAGS"] = os.environ.get(
        "NEURON_CC_FLAGS",
        "--model-type transformer --retry_failed_compilation"
    )
    os.environ["NEURON_FUSE_SOFTMAX"] = "1"
    os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "3"
    os.environ["MALLOC_ARENA_MAX"] = "64"  # Host OOM mitigation

    log.info("Neuron environment configured")
    log.info(f"NEURON_CC_FLAGS: {os.environ.get('NEURON_CC_FLAGS')}")

    # Display configuration
    if get_global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)

        # Save config
        if not cfg.dry_run:
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            log.info(f"Saving config to {save_path}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            cfg.save(save_path)

    barrier()

    # Set seed
    seed_all(cfg.seed)

    # Calculate device batch sizes (required before building data loader)
    from olmo.torch_util import get_world_size
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    log.info(f"Batch sizes: global={cfg.global_train_batch_size}, device={cfg.device_train_batch_size}, "
             f"microbatch={cfg.device_train_microbatch_size}, grad_accum={cfg.device_train_grad_accum}")

    # Build data loader (using OLMo's custom data loader)
    log.info("Building data loader...")
    train_loader = build_train_dataloader(cfg)

    # Build evaluators (if any)
    evaluators = build_evaluators(cfg, torch.device("cpu"))  # Device doesn't matter for neuron
    barrier()

    # Initialize neuronx_distributed parallel groups before creating NeuronTrainingArguments
    # Note: OLMo is not a custom neuron modeling model, so we can only use data parallelism
    log.info("Initializing neuronx_distributed parallel groups...")
    from neuronx_distributed.parallel_layers import parallel_state
    # Use tensor_parallel_size=1 for non-optimum models like OLMo
    tensor_parallel_size = 1
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)
    log.info(f"Parallel groups initialized with tensor_model_parallel_size={tensor_parallel_size} (data parallel only)")

    # Convert OLMo config to Neuron training arguments
    log.info("Creating Neuron training configuration...")
    neuron_args = convert_olmo_to_neuron_config(cfg)

    # Initialize model for Neuron
    log.info("Building OLMo model for Neuron...")

    # Set precision
    dtype = torch.bfloat16 if cfg.precision == "amp_bf16" else (
        torch.float16 if cfg.precision == "amp_fp16" else torch.float32
    )

    # Build OLMo model with neuron wrapper for optimum-neuron compatibility
    # This wrapper makes OLMo appear as a custom neuron modeling model
    log.info("Creating OLMo model with neuron wrapper...")
    olmo_model = OLMoForCausalLM(cfg.model, trn_config=None)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

    # Convert to dtype
    olmo_model = olmo_model.to(dtype)

    # Note: optimum-neuron will handle the parallelization and compilation
    # We don't need to manually wrap with FSDP or move to device

    log.info("Model initialized")
    log.info(f"Model dtype: {dtype}")

    # Initialize W&B if configured
    if cfg.wandb is not None and get_global_rank() == 0:
        import wandb
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=str(wandb_dir),
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name or cfg.run_name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )
        log.info("Weights & Biases initialized")

    # Create trainer
    log.info("Creating OLMoNeuronTrainer...")
    trainer = OLMoNeuronTrainer(
        olmo_config=cfg,
        train_loader=train_loader,
        model=olmo_model,
        args=neuron_args,
        train_dataset=train_loader.dataset if hasattr(train_loader, 'dataset') else None,
    )

    # Load checkpoint if specified
    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        # optimum-neuron handles checkpoint loading
        trainer.train(resume_from_checkpoint=cfg.load_path)
    else:
        # Start training
        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.train()
            log.info("Training complete")

            # Save final model
            log.info("Saving final model...")
            trainer.save_model()
            log.info(f"Model saved to {cfg.save_folder}")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    import torch.distributed as dist
    from datetime import timedelta

    # Initialize distributed training (required for Neuron)
    if not os.getenv("RANK"):
        os.environ["RANK"] = "0"
    if not os.getenv("WORLD_SIZE"):
        os.environ["WORLD_SIZE"] = "1"
    if not os.getenv("MASTER_ADDR"):
        os.environ["MASTER_ADDR"] = "localhost"
    if not os.getenv("MASTER_PORT"):
        os.environ["MASTER_PORT"] = "24501"

    # Initialize process group for Neuron
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=30))
    log.info("Process group initialized")

    # Prepare environment
    prepare_cli_environment()

    # Parse arguments
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoConfigurationError("Usage: train-optimum-neuron.py <config.yaml> [--key=value ...]")

    # Load config
    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])

    # Run training
    main(cfg)
