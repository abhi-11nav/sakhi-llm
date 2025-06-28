import json
import os
import time
from datetime import datetime

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import wandb
from torch.amp import GradScaler, autocast
from torch.distributed import destroy_process_group
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from sakhilabs.configs.utils.config import SakhiConfig
from sakhilabs.data.custom_dataset.pretraining.custom_dataset import \
    CustomDataset
from sakhilabs.pipelines.utils.general_utils import (do_sanity_checks,
                                                     get_sakhi_model, setup,
                                                     setup_logging)
from sakhilabs.pipelines.utils.training_utils import hash_tensor, set_seed


def train(
    rank: int,
    world_size: int,
    config: SakhiConfig,
):
    try:
        log_dir = config.paths.log_dir
        logger = setup_logging(rank, log_dir=log_dir)
        logger.info(f"Starting DDP training on rank {rank}")

        save_every_n_steps = config.train_parameters.save_every_n_steps

        training_data = {
            "config": {
                "rank": rank,
                "world_size": world_size,
                "embed_dim": config.model_parameters.embed_dim,
                "num_heads": config.model_parameters.num_heads,
                "ff_dim": config.model_parameters.ff_dim,
                "chunk_length": config.model_parameters.chunk_length,
                "num_layers": config.model_parameters.num_layers,
                "batch_size": config.train_parameters.batch_size,
                "num_epochs": config.train_parameters.num_epochs,
                "learning_rate": config.train_parameters.init_learning_rate,
                "jsonl_path": config.paths.dataset_path,
                "vocab_size": config.model_parameters.vocab_size,
            },
            "training_progress": [],
            "epoch_summaries": [],
            "model_saves": [],
        }

        setup(rank, world_size, config)
        logger.info(f"Process group initialized for rank {rank}")

        # Create dataset
        logger.info("Creating dataset")
        dataset = CustomDataset(
            config.paths.dataset_path,
            chunk_length=config.model_parameters.chunk_length,
        )

        logger.info(f"Vocabulary size: {config.model_parameters.vocab_size}")

        # Create model and move to GPU
        logger.info("Initializing Sakhi model...")
        sakhi_model = get_sakhi_model(rank=rank, world_size=world_size, config=config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.logging.wandb:
            wandb.init(
                project="Sakhi-Model-Training",
                config={
                    "epochs": config.train_parameters.num_epochs,
                    "batch_size": config.train_parameters.batch_size,
                    "learning_rate": config.train_parameters.init_learning_rate,
                    "model_params": {
                        "embed_dim": config.model_parameters.embed_dim,
                        "num_heads": config.model_parameters.num_heads,
                        "ff_dim": config.model_parameters.ff_dim,
                        "num_layers": config.model_parameters.num_layers,
                        "vocab_size": config.model_parameters.vocab_size,
                    },
                    "world_size": world_size,
                },
                group="distributed_training_run",
                job_type=f"rank_{rank}",
                name=f"soki_train_rank_{rank}_{timestamp}",
                reinit=True,
            )

        if rank == 0:
            model_ref = sakhi_model.module if world_size > 1 else sakhi_model
            total_params = sum(p.numel() for p in model_ref.parameters())
            logger.info(f"Model initialized with {total_params:,} parameters")
            training_data["config"]["total_parameters"] = total_params

        logger.info(
            f"Initializing DataLoader with batch_size={config.train_parameters.batch_size}"
        )
        data_loader = DataLoader(
            dataset,
            batch_size=config.train_parameters.batch_size,
            num_workers=config.data_loader.num_workers,
            pin_memory=config.data_loader.pin_memory,
        )

        # Loss, Optimizer and LRScheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            sakhi_model.parameters(),
            lr=float(config.train_parameters.init_learning_rate),
        )

        # After optimizer creation:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(data_loader),
            eta_min=float(config.train_parameters.min_learning_rate),
        )

        logger.info("Starting training loop...")
        training_start_time = time.time()

        scaler = GradScaler()

        # Training loop
        for epoch in range(config.train_parameters.num_epochs):
            epoch_start_time = time.time()
            logger.info(
                f"Starting epoch {epoch + 1}/{config.train_parameters.num_epochs}"
            )

            epoch_loss = 0.0
            num_batches = 0
            batch_losses = []
            grad_accum_steps = config.train_parameters.gradient_accumulation_steps

            if rank == 0:
                batch_iterator = tqdm(
                    enumerate(data_loader),
                    total=len(data_loader),
                    desc=f"Epoch {epoch + 1}",
                )
            else:
                batch_iterator = enumerate(data_loader)

            optimizer.zero_grad()

            for i, batch in batch_iterator:
                batch_start_time = time.time()

                input_ids = batch["input_ids"].to(rank, non_blocking=True)
                labels = batch["labels"].to(rank, non_blocking=True)

                with autocast(device_type="cuda"):
                    output_logits = sakhi_model(input_ids)
                    loss = criterion(
                        output_logits.view(-1, config.model_parameters.vocab_size),
                        labels.reshape(-1),
                    )

                    loss = loss / grad_accum_steps
                # Backward pass
                scaler.scale(loss).backward()

                # Gradient accumulation
                if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        sakhi_model.parameters(),
                        max_norm=config.train_parameters.gradient_clipping_max_norm,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    scheduler.step()

                batch_time = time.time() - batch_start_time
                loss_value = loss.item()
                epoch_loss += loss_value
                num_batches += 1
                batch_losses.append(loss_value)

                if i < 10:
                    local_rank = rank
                    token_hash = hash_tensor(input_ids[0])
                    logger.info(
                        f"Dataset Duplication Info:: Rank {local_rank}, Step {i}, InputHash: {token_hash}"
                    )

                if i % config.train_parameters.log_every_n_steps == 0:
                    if config.logging.wandb:
                        wandb.log(
                            {
                                "epoch": epoch + 1,
                                "step": i + epoch * len(data_loader),
                                "batch_loss": loss_value,
                                "batch_time": batch_time,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "rank": rank,
                            }
                        )

                    if rank == 0:
                        logger.info(
                            f"Epoch {epoch + 1}/{config.train_parameters.num_epochs}, Batch {i}, "
                            f"Loss: {loss_value:.4f}, Batch Time: {batch_time:.2f}s"
                        )

                        # store to training_data dictionary
                        training_data["training_progress"].append(
                            {
                                "epoch": epoch + 1,
                                "batch": i,
                                "loss": loss_value,
                                "batch_time": batch_time,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        if len(training_data["training_progress"]) > 3000:
                            training_data["training_progress"] = training_data[
                                "training_progress"
                            ][-3000:]

                # Save model at n steps in each epoch
                if (
                    rank == 0
                    and save_every_n_steps
                    and i > 0
                    and i % save_every_n_steps == 0
                ):
                    model_save_dir = config.paths.model_dir
                    model_filename = (
                        f"{model_save_dir}/soki_model_epoch_{epoch + 1}_step{i}.pth"
                    )
                    state_dict = (
                        sakhi_model.module.state_dict()
                        if world_size > 1
                        else sakhi_model.state_dict()
                    )
                    torch.save(state_dict, model_filename)
                    logger.info(f"Model saved to {model_filename}")

                    training_data["model_saves"].append(
                        {
                            "epoch": epoch + 1,
                            "step": i,
                            "filename": model_filename,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                    json_filename = os.path.join(
                        log_dir, f"training_data_rank_{rank}.json"
                    )
                    with open(json_filename, "w") as f:
                        json.dump(training_data, f, indent=2)

            epoch_time = time.time() - epoch_start_time

            # Calculate and log epoch summary
            if rank == 0:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
                min_loss = min(batch_losses) if batch_losses else 0
                max_loss = max(batch_losses) if batch_losses else 0

                logger.info(f"EPOCH {epoch + 1} SUMMARY")
                logger.info(f"Average Loss: {avg_loss:.4f}")
                logger.info(f"Min Loss: {min_loss:.4f}")
                logger.info(f"Max Loss: {max_loss:.4f}")
                logger.info(f"Total Batches: {num_batches}")
                logger.info(f"Epoch Time: {epoch_time:.2f}s")
                logger.info(f"Batches/sec: {num_batches / epoch_time:.2f}\n")

                # Store epoch summary for JSON
                epoch_summary = {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "total_batches": num_batches,
                    "epoch_time": epoch_time,
                    "batches_per_sec": num_batches / epoch_time,
                    "timestamp": datetime.now().isoformat(),
                }
                training_data["epoch_summaries"].append(epoch_summary)

            # Empty cache after each epoch
            torch.cuda.empty_cache()
        total_training_time = time.time() - training_start_time

        if config.logging.wandb:
            wandb.finish()

        # Final logging
        if rank == 0:
            logger.info("Training Completed Succesfully")
            logger.info(f"Total Training Time: {total_training_time:.2f}s")
            logger.info(
                f"Average Time per Epoch: {total_training_time / config.train_parameters.num_epochs:.2f}s\n"
            )

            training_data["final_summary"] = {
                "total_training_time": total_training_time,
                "avg_time_per_epoch": total_training_time
                / config.train_parameters.num_epochs,
                "completion_timestamp": datetime.now().isoformat(),
            }

            # Save final json
            final_json_filename = os.path.join(
                log_dir, f"final_training_data_rank_{rank}.json"
            )
            with open(final_json_filename, "w") as f:
                json.dump(training_data, f, indent=2)
            logger.info(f"Training data saved to {final_json_filename}")

        # Save final model
        if rank == 0:
            model_save_dir = config.paths.model_dir
            final_model_filename = f"{model_save_dir}/soki_model_final.pth"
            state_dict = (
                sakhi_model.module.state_dict()
                if world_size > 1
                else sakhi_model.state_dict()
            )
            torch.save(state_dict, final_model_filename)
            logger.info(f"Final model saved to {final_model_filename}")

        logger.info(f"Rank {rank} training completed. Cleaning up...")
    except Exception as e:
        logger.error(f"Error occured while training {e}")
        raise
    finally:
        if world_size > 1:
            destroy_process_group()


def pretraining_run(config: SakhiConfig):
    # do sanity checks and set seed
    do_sanity_checks(config=config)
    set_seed(seed=config.train_parameters.seed)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)
    world_size = (
        torch.cuda.device_count()
        if config.train_parameters.num_gpus == -1
        else config.train_parameters.num_gpus
    )

    config.model_parameters.vocab_size = len(tokenizer)
    del tokenizer

    if world_size > 1:
        # DDP
        mp.spawn(
            train,
            args=(world_size, config),
            nprocs=world_size,
            join=True,
        )
    else:
        train(rank=0, world_size=world_size, config=config)
