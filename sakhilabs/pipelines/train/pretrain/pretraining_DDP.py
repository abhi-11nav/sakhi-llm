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

from sakhilabs.configs.utils.load_config import SakhiConfig
from sakhilabs.pipelines.train.pretrain.dataset import SakhiPreTrainDataset
from sakhilabs.pipelines.utils.constants import TrainMode
from sakhilabs.pipelines.utils.cook_model import get_sakhi_model
from sakhilabs.pipelines.utils.general_utils import (do_sanity_checks, setup,
                                                     setup_logging)
from sakhilabs.pipelines.utils.training_utils import hash_tensor, set_seed

import glob


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
                "min_learning_rate": config.train_parameters.min_learning_rate,
                "gradient_accumulation_steps": config.train_parameters.gradient_accumulation_steps,
                "gradient_clipping_max_norm": config.train_parameters.gradient_clipping_max_norm,
                "jsonl_path": config.paths.dataset_path,
                "vocab_size": config.model_parameters.vocab_size,
                "save_every_n_steps": save_every_n_steps,
                "log_every_n_steps": config.train_parameters.log_every_n_steps,
            },
            "training_progress": [],
            "epoch_summaries": [],
            "model_saves": [],
            "detailed_metrics": {
                "losses": [],
                "learning_rates": [],
                "batch_times": [],
                "epoch_metrics": [],
                "file_metrics": []
            }
        }

        setup(rank, world_size, config)
        logger.info(f"Process group initialized for rank {rank}")

        # Create dataset
        logger.info(f"Vocabulary size: {config.model_parameters.vocab_size}")
        logger.info("Initializing Sakhi model...")

        sakhi_model = get_sakhi_model(
            embed_dim=config.model_parameters.embed_dim,
            num_heads=config.model_parameters.num_heads,
            ff_dim=config.model_parameters.ff_dim,
            vocab_size=config.model_parameters.vocab_size,
            num_layers=config.model_parameters.num_layers,
            rank=rank,
            world_size=world_size,
            train_mode=TrainMode.DDP,
            resume=config.train_parameters.resume,
            resize_model_output_to_size=config.model_parameters.vocab_size,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if config.logger.wandb:
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

        # Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            sakhi_model.parameters(),
            lr=float(config.train_parameters.init_learning_rate),
        )

        logger.info("Starting training loop...")
        training_start_time = time.time()

        scaler = GradScaler()
        dataset_folder = config.paths.dataset_path
        dataset_files = sorted(glob.glob(os.path.join(dataset_folder, "*")))
        
        # Calculate total steps for scheduler
        total_steps_per_epoch = 0
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)
        
        for file_path in dataset_files:
            temp_dataset = SakhiPreTrainDataset(
                dataset_json=file_path,
                chunk_length=config.model_parameters.chunk_length,
                tokenizer=tokenizer
            )
            temp_loader = DataLoader(
                temp_dataset,
                batch_size=config.train_parameters.batch_size,
                num_workers=config.data_loader.num_workers,
                pin_memory=config.data_loader.pin_memory,
            )
            total_steps_per_epoch += len(temp_loader)
            del temp_dataset, temp_loader
        
        del tokenizer
        
        # Initialize scheduler with total steps across all files and epochs
        total_training_steps = total_steps_per_epoch * config.train_parameters.num_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_training_steps,
            eta_min=float(config.train_parameters.min_learning_rate),
        )
        
        logger.info(f"Total steps per epoch: {total_steps_per_epoch}")
        logger.info(f"Total training steps: {total_training_steps}")

        # Global tracking variables
        global_step = 0
        global_sample_index = 0
        
        # Metrics tracking for plotting
        all_losses = []
        all_learning_rates = []
        all_batch_times = []
        all_global_steps = []
        
        # Training loop
        for epoch in range(config.train_parameters.num_epochs):
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{config.train_parameters.num_epochs}")

            # Epoch-level metrics (accumulate across all files)
            epoch_loss = 0.0
            epoch_num_batches = 0
            epoch_batch_losses = []
            epoch_learning_rates = []
            epoch_batch_times = []
            
            for file_idx, each_datafile in enumerate(dataset_files):
                file_start_time = time.time()
                logger.info(f"Processing file {file_idx + 1}/{len(dataset_files)}: {os.path.basename(each_datafile)}")
                
                # File-level metrics
                file_loss = 0.0
                file_num_batches = 0
                file_batch_losses = []
                
                # Recreate tokenizer for each file (since you delete it)
                tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)
                
                dataset = SakhiPreTrainDataset(
                    dataset_json=each_datafile,
                    chunk_length=config.model_parameters.chunk_length,
                    tokenizer=tokenizer
                )

                data_loader = DataLoader(
                    dataset,
                    batch_size=config.train_parameters.batch_size,
                    num_workers=config.data_loader.num_workers,
                    pin_memory=config.data_loader.pin_memory,
                )

                grad_accum_steps = config.train_parameters.gradient_accumulation_steps

                if rank == 0:
                    batch_iterator = tqdm(
                        enumerate(data_loader),
                        total=len(data_loader),
                        desc=f"Epoch {epoch + 1}, File {file_idx + 1}/{len(dataset_files)}",
                    )
                else:
                    batch_iterator = enumerate(data_loader)

                for i, batch in batch_iterator:
                    batch_start_time = time.time()

                    input_ids = batch["input_ids"].to(rank, non_blocking=True)
                    labels = batch["labels"].to(rank, non_blocking=True)

                    actual_batch_size = input_ids.size(0)

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

                    global_sample_index += actual_batch_size
                    global_step += 1

                    # Save checkpoint based on global steps
                    if global_step % save_every_n_steps == 0 and rank == 0:
                        with open(
                            os.path.join(config.paths.save_dir, "start_sample.json"), "w"
                        ) as f:
                            json.dump({"start_sample": global_sample_index, "global_step": global_step}, f)

                    batch_time = time.time() - batch_start_time
                    loss_value = loss.item()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Accumulate epoch metrics
                    epoch_loss += loss_value
                    epoch_num_batches += 1
                    epoch_batch_losses.append(loss_value)
                    epoch_learning_rates.append(current_lr)
                    epoch_batch_times.append(batch_time)
                    
                    # Accumulate file metrics
                    file_loss += loss_value
                    file_num_batches += 1
                    file_batch_losses.append(loss_value)
                    
                    # Accumulate global metrics for plotting
                    all_losses.append(loss_value)
                    all_learning_rates.append(current_lr)
                    all_batch_times.append(batch_time)
                    all_global_steps.append(global_step)

                    if i < 10:
                        local_rank = rank
                        token_hash = hash_tensor(input_ids[0])
                        logger.info(
                            f"Dataset Duplication Info:: Rank {local_rank}, Global Step {global_step}, File {file_idx + 1}, Batch {i}, InputHash: {token_hash}"
                        )

                    if global_step % config.train_parameters.log_every_n_steps == 0:
                        if config.logger.wandb:
                            wandb.log(
                                {
                                    "epoch": epoch + 1,
                                    "global_step": global_step,
                                    "file_index": file_idx,
                                    "batch_in_file": i,
                                    "batch_loss": loss_value,
                                    "batch_time": batch_time,
                                    "learning_rate": current_lr,
                                    "rank": rank,
                                }
                            )

                        if rank == 0:
                            logger.info(
                                f"Epoch {epoch + 1}/{config.train_parameters.num_epochs}, "
                                f"File {file_idx + 1}/{len(dataset_files)}, Batch {i}, "
                                f"Global Step {global_step}, Loss: {loss_value:.4f}, "
                                f"LR: {current_lr:.2e}, Batch Time: {batch_time:.2f}s"
                            )

                            # Store to training_data dictionary
                            training_data["training_progress"].append(
                                {
                                    "epoch": epoch + 1,
                                    "global_step": global_step,
                                    "file_index": file_idx,
                                    "batch_in_file": i,
                                    "loss": loss_value,
                                    "learning_rate": current_lr,
                                    "batch_time": batch_time,
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )

                            if len(training_data["training_progress"]) > 3000:
                                training_data["training_progress"] = training_data[
                                    "training_progress"
                                ][-3000:]

                    # Save model based on global steps
                    if (
                        rank == 0
                        and save_every_n_steps
                        and global_step > 0
                        and global_step % save_every_n_steps == 0
                    ):
                        model_save_dir = config.paths.model_dir
                        model_filename = (
                            f"{model_save_dir}/soki_model_epoch_{epoch + 1}_step_{global_step}.pth"
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
                                "global_step": global_step,
                                "filename": model_filename,
                                "avg_loss_last_100": sum(all_losses[-100:]) / min(len(all_losses), 100),
                                "current_lr": current_lr,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        # Save incremental metrics after each checkpoint
                        incremental_json_filename = os.path.join(
                            log_dir, f"training_metrics_rank_{rank}.json"
                        )
                        
                        # Update detailed metrics
                        training_data["detailed_metrics"]["losses"] = all_losses[-1000:]  # Keep last 1000
                        training_data["detailed_metrics"]["learning_rates"] = all_learning_rates[-1000:]
                        training_data["detailed_metrics"]["batch_times"] = all_batch_times[-1000:]
                        
                        with open(incremental_json_filename, "w") as f:
                            json.dump(training_data, f, indent=2)

                file_time = time.time() - file_start_time
                
                # Save file-level metrics
                if rank == 0:
                    file_avg_loss = file_loss / file_num_batches if file_num_batches > 0 else 0
                    file_min_loss = min(file_batch_losses) if file_batch_losses else 0
                    file_max_loss = max(file_batch_losses) if file_batch_losses else 0
                    
                    file_metrics = {
                        "epoch": epoch + 1,
                        "file_index": file_idx,
                        "file_name": os.path.basename(each_datafile),
                        "avg_loss": file_avg_loss,
                        "min_loss": file_min_loss,
                        "max_loss": file_max_loss,
                        "num_batches": file_num_batches,
                        "file_time": file_time,
                        "batches_per_sec": file_num_batches / file_time if file_time > 0 else 0,
                        "global_steps_start": global_step - file_num_batches + 1,
                        "global_steps_end": global_step,
                        "timestamp": datetime.now().isoformat(),
                    }
                    training_data["detailed_metrics"]["file_metrics"].append(file_metrics)
                    
                    logger.info(f"File {file_idx + 1} completed - Avg Loss: {file_avg_loss:.4f}, "
                              f"Batches: {file_num_batches}, Time: {file_time:.2f}s")

                # Clean up after each file
                del dataset, data_loader, tokenizer
                torch.cuda.empty_cache()

            epoch_time = time.time() - epoch_start_time

            # Calculate and log epoch summary (accumulated across all files)
            if rank == 0:
                avg_loss = epoch_loss / epoch_num_batches if epoch_num_batches > 0 else 0
                min_loss = min(epoch_batch_losses) if epoch_batch_losses else 0
                max_loss = max(epoch_batch_losses) if epoch_batch_losses else 0
                avg_lr = sum(epoch_learning_rates) / len(epoch_learning_rates) if epoch_learning_rates else 0
                avg_batch_time = sum(epoch_batch_times) / len(epoch_batch_times) if epoch_batch_times else 0

                logger.info(f"EPOCH {epoch + 1} SUMMARY")
                logger.info(f"Average Loss: {avg_loss:.4f}")
                logger.info(f"Min Loss: {min_loss:.4f}")
                logger.info(f"Max Loss: {max_loss:.4f}")
                logger.info(f"Average LR: {avg_lr:.2e}")
                logger.info(f"Average Batch Time: {avg_batch_time:.3f}s")
                logger.info(f"Total Batches: {epoch_num_batches}")
                logger.info(f"Total Files Processed: {len(dataset_files)}")
                logger.info(f"Epoch Time: {epoch_time:.2f}s")
                logger.info(f"Batches/sec: {epoch_num_batches / epoch_time:.2f}")
                logger.info(f"Global Steps Completed: {global_step}\n")

                # Store epoch summary for JSON
                epoch_summary = {
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "min_loss": min_loss,
                    "max_loss": max_loss,
                    "loss_std": float(torch.tensor(epoch_batch_losses).std().item()) if epoch_batch_losses else 0,
                    "avg_learning_rate": avg_lr,
                    "min_learning_rate": min(epoch_learning_rates) if epoch_learning_rates else 0,
                    "max_learning_rate": max(epoch_learning_rates) if epoch_learning_rates else 0,
                    "avg_batch_time": avg_batch_time,
                    "total_batches": epoch_num_batches,
                    "total_files": len(dataset_files),
                    "epoch_time": epoch_time,
                    "batches_per_sec": epoch_num_batches / epoch_time,
                    "global_steps_completed": global_step,
                    "samples_processed": global_sample_index,
                    "timestamp": datetime.now().isoformat(),
                }
                training_data["epoch_summaries"].append(epoch_summary)
                training_data["detailed_metrics"]["epoch_metrics"].append(epoch_summary)

                # Save comprehensive metrics after each epoch
                epoch_metrics_filename = os.path.join(
                    log_dir, f"epoch_{epoch + 1}_metrics_rank_{rank}.json"
                )
                epoch_data = {
                    "epoch_summary": epoch_summary,
                    "epoch_losses": epoch_batch_losses,
                    "epoch_learning_rates": epoch_learning_rates,
                    "epoch_batch_times": epoch_batch_times,
                    "file_metrics": training_data["detailed_metrics"]["file_metrics"][-len(dataset_files):],
                    "model_saves_this_epoch": [
                        save for save in training_data["model_saves"] 
                        if save["epoch"] == epoch + 1
                    ]
                }
                with open(epoch_metrics_filename, "w") as f:
                    json.dump(epoch_data, f, indent=2)
                
                logger.info(f"Epoch {epoch + 1} metrics saved to {epoch_metrics_filename}")

            # Empty cache after each epoch
            torch.cuda.empty_cache()
            
        total_training_time = time.time() - training_start_time

        if config.logger.wandb:
            wandb.finish()

        # Final logging and comprehensive metrics save
        if rank == 0:
            logger.info("Training Completed Successfully")
            logger.info(f"Total Training Time: {total_training_time:.2f}s")
            logger.info(f"Total Global Steps: {global_step}")
            logger.info(
                f"Average Time per Epoch: {total_training_time / config.train_parameters.num_epochs:.2f}s\n"
            )

            # Calculate final statistics
            final_avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0
            final_min_loss = min(all_losses) if all_losses else 0
            final_max_loss = max(all_losses) if all_losses else 0
            final_loss_std = float(torch.tensor(all_losses).std().item()) if all_losses else 0

            training_data["final_summary"] = {
                "total_training_time": total_training_time,
                "total_global_steps": global_step,
                "total_samples_processed": global_sample_index,
                "avg_time_per_epoch": total_training_time / config.train_parameters.num_epochs,
                "overall_avg_loss": final_avg_loss,
                "overall_min_loss": final_min_loss,
                "overall_max_loss": final_max_loss,
                "overall_loss_std": final_loss_std,
                "total_files_processed": len(dataset_files) * config.train_parameters.num_epochs,
                "completion_timestamp": datetime.now().isoformat(),
            }

            # Save final comprehensive metrics
            training_data["detailed_metrics"]["losses"] = all_losses
            training_data["detailed_metrics"]["learning_rates"] = all_learning_rates
            training_data["detailed_metrics"]["batch_times"] = all_batch_times
            training_data["detailed_metrics"]["global_steps"] = all_global_steps

            # Save final json with all metrics
            final_json_filename = os.path.join(
                log_dir, f"final_training_data_rank_{rank}.json"
            )
            with open(final_json_filename, "w") as f:
                json.dump(training_data, f, indent=2)
            logger.info(f"Final training data saved to {final_json_filename}")

            # Save a separate plotting-friendly JSON
            plotting_data = {
                "config": training_data["config"],
                "global_steps": all_global_steps,
                "losses": all_losses,
                "learning_rates": all_learning_rates,
                "batch_times": all_batch_times,
                "epoch_summaries": training_data["epoch_summaries"],
                "model_saves": training_data["model_saves"],
                "final_summary": training_data["final_summary"],
            }
            
            plotting_json_filename = os.path.join(
                log_dir, f"plotting_data_rank_{rank}.json"
            )
            with open(plotting_json_filename, "w") as f:
                json.dump(plotting_data, f, indent=2)
            logger.info(f"Plotting data saved to {plotting_json_filename}")

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
        logger.error(f"Error occurred while training {e}")
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
    os.environ["WANDB_MODE"] = config.logger.mode

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


if __name__ == "__main__":
    config_path = "sakhilabs/configs/sakhi-telugu-681M-pretrained-0625.yaml"
    config = SakhiConfig._load_config(config_path=config_path)
    pretraining_run(config=config)
