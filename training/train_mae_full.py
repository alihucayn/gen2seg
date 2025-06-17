# gen2seg official inference pipeline code for Stable Diffusion model
# 
# Please see our project website at https://reachomk.github.io/gen2seg
#
# Additionally, if you use our code please cite our paper. 

import argparse
import logging
import math
import os
import shutil
import gc

import accelerate
import datasets
import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, ViTMAEForPreTraining, AutoConfig
from einops import rearrange

import diffusers
from diffusers.utils import check_min_version, is_wandb_available

from torch.optim.lr_scheduler import LambdaLR

# Custom modules (assumed to be in your PYTHONPATH)
from dataloaders.load import *
from util.loss import InstanceSegmentationLoss
from util.lr_scheduler import IterExponential

if is_wandb_available():
    import wandb

# Ensure the minimal version of diffusers is installed.
check_min_version("0.27.0.dev0")
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training code for fine-tuning ViTMAEForPreTraining with instance loss."
    )

    parser.add_argument("--lr_exp_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_total_iter_length", type=int, default=20000)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files (e.g., fp16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training/model-finetuned",
        required=True,
        help="The output directory for saving checkpoints and the final model.",
    )
    parser.add_argument("--seed", type=int, default=500, help="Seed for reproducibility.")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size per device.")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        required=True,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Number of steps to accumulate gradients before updating.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1 parameter.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2 parameter.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20000,
        help="Save a checkpoint every X steps.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Maximum number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to resume from a previous checkpoint.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Enable xformers memory efficient attention.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="e2e-ft-diffusion",
        help="Project name for tracking.",
    )
    parser.add_argument(
        "--random_state_file",
        type=str,
        default=None,
        help="Path to a random state file for reproducibility.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    args = parse_args()
    # Initialize Accelerator and logger.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set seed for reproducibility.
    if args.seed is not None:
        set_seed(args.seed)
    if args.random_state_file is not None:
        logger.info(f"Loading random states from file {args.random_state_file}")
        random_states = torch.load(args.random_state_file)
        import random
        import numpy as np
        random.setstate(random_states["random_state"])
        np.random.set_state(random_states["numpy_random_seed"])
        torch.set_rng_state(random_states["torch_manual_seed"])
        if "torch_cuda_manual_seed" in random_states:
            for i, cuda_state in enumerate(random_states["torch_cuda_manual_seed"]):
                if torch.cuda.device_count() > i:
                    torch.cuda.set_rng_state(cuda_state, device=i)

    # Save training arguments.
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "arguments.txt"), "w") as f:
            args_dict = vars(args)
            f.write("\n".join(f"{k}: {v}" for k, v in args_dict.items()))

    # -------------------------------
    # Model: Load ViTMAEForPreTraining and set mask_ratio to 0.
    # -------------------------------
    logger.info("Loading ViTMAEForPreTraining model")
    image_size = 224  # e.g., 224

    config = AutoConfig.from_pretrained(args.pretrained_model_name_or_path, revision=args.revision, image_size=image_size)

    model = ViTMAEForPreTraining.from_pretrained(args.pretrained_model_name_or_path, config=config, ignore_mismatched_sizes=True)
    patch_size = model.config.patch_size  # e.g., 16
    model.config.mask_ratio = 0.0  # disable masking for finetuning
    model.train()
    model = model.cuda()
    image_processor = AutoImageProcessor.from_pretrained(args.pretrained_model_name_or_path)
    image_processor.size = image_size

    # -------------------------------
    # Optimizer and Learning Rate Scheduler
    # -------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_func = IterExponential(
        total_iter_length=args.lr_total_iter_length * accelerator.num_processes,
        final_ratio=0.05,
        warmup_steps=args.lr_exp_warmup_steps * accelerator.num_processes,
    )
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # -------------------------------
    # Setup Datasets and Dataloaders
    # -------------------------------
    vkitti_root_dir = "UPDATE THIS"
    hypersim_root_dir = "UPDATE THIS"
    train_dataset_hypersim = Hypersim(root_dir=hypersim_root_dir, transform=True, height=224, width=224)
    train_dataset_vkitti = VirtualKITTI2(root_dir=vkitti_root_dir,res=(224, 224), transform=True)

    train_dataloader_vkitti = torch.utils.data.DataLoader(
        train_dataset_vkitti,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader_hypersim = torch.utils.data.DataLoader(
        train_dataset_hypersim,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Custom mixed dataloader combining both datasets.
    train_dataloader = MixedDataLoader(
        train_dataloader_hypersim,
        train_dataloader_vkitti,
        split1=9,
        split2=1,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        model.to(dtype=weight_dtype)
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        model.to(dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_vkitti) + len(train_dataset_hypersim)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting new.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    # -------------------------------
    # Training Loop
    # -------------------------------
    instance_loss = InstanceSegmentationLoss()

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f"Epoch {epoch} / {args.num_train_epochs}")
        torch.cuda.empty_cache()
        gc.collect()

        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Preprocess the input images.
                images0_1 = (batch["rgb"] + 1) / 2
                images0_1 = images0_1.cuda()
                transformer_input = image_processor(images=images0_1, return_tensors="pt", do_rescale=False)
                transformer_input = {k: v.to(accelerator.device) for k, v in transformer_input.items()}

                # Forward pass.
                outputs = model(pixel_values=transformer_input["pixel_values"])
                logits = outputs.logits  # shape: (B, num_patches, patch_dim)


                grid_size = image_size // patch_size

                # Reshape the logits to reconstructed images of shape (B, 3, H, W)
                reconstructed = rearrange(
                    logits,
                    "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                    h=grid_size,
                    p1=patch_size,
                    p2=patch_size,
                    c=3
                )
                current_estimate = reconstructed

                # Normalize reconstructed images to [0, 255]
                min_val = torch.abs(current_estimate.min())
                max_val = torch.abs(current_estimate.max())
                current_estimate = (current_estimate + min_val) / (max_val + min_val + 1e-5)
                current_estimate = current_estimate * 255.0

                ground_truth = batch["instance"].to(device=accelerator.device, dtype=weight_dtype)
                loss = instance_loss(current_estimate, ground_truth, batch["no_bg"])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                avg_loss = accelerator.gather(loss.detach()).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    accelerator.log({"lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                    train_loss = 0.0

                    if global_step % args.checkpointing_steps == 0:
                        logger.info(f"Saving checkpoint at step {global_step}")
                        if accelerator.is_main_process:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[:num_to_remove]
                                    logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                                    for rm_ckpt in removing_checkpoints:
                                        shutil.rmtree(os.path.join(args.output_dir, rm_ckpt))
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                if global_step >= args.max_train_steps:
                    break

        if global_step >= args.max_train_steps:
            break

    # -------------------------------
    # Post-training: Save the full model.
    # -------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model_to_save = accelerator.unwrap_model(model)
        logger.info(f"Saving the full fine-tuned ViTMAEForPreTraining model to {args.output_dir}")
        model_to_save.save_pretrained(args.output_dir)

    logger.info("Finished training.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
