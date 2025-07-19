import os
import uuid
import json
import yaml
import shutil
import subprocess
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from PIL import Image
import torch

logger = logging.getLogger(__name__)

class TrainingManager:
    """Manager class for the entire training pipeline"""
    
    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.training_status: Dict[str, Dict] = {}
        
    # ... (các hàm create_training_session, update_status, get_status, save_variations, create_dataset_config, create_training_config không thay đổi)
    def create_training_session(self, name_object: str, description: Optional[str] = None) -> str:
        """Create a new session for training"""
        training_id = str(uuid.uuid4())
        session_dir = self.base_dir / training_id
        session_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (session_dir / "variations").mkdir(exist_ok=True)
        (session_dir / "dataset").mkdir(exist_ok=True)
        (session_dir / "model").mkdir(exist_ok=True)
        (session_dir / "output").mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            "training_id": training_id,
            "name_object": name_object,
            "description": description,
            "created_at": str(Path().absolute()),
            "status": "initialized"
        }
        
        with open(session_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        self.training_status[training_id] = {
            "status": "initialized",
            "progress": 0.0,
            "message": "Training session created"
        }
        
        return training_id
    
    def update_status(self, training_id: str, status: str, progress: float, message: str):
        """Update status of training session"""
        if training_id not in self.training_status:
            self.training_status[training_id] = {}
            
        self.training_status[training_id].update({
            "status": status,
            "progress": progress,
            "message": message
        })
        
        # Save to file
        session_dir = self.base_dir / training_id
        if session_dir.exists():
            with open(session_dir / "status.json", "w") as f:
                json.dump(self.training_status[training_id], f, indent=2)
    
    def get_status(self, training_id: str) -> Dict:
        """Get status of training session"""
        return self.training_status.get(training_id, {
            "status": "not_found",
            "progress": 0.0,
            "message": "Training session not found"
        })
    
    def save_variations(self, training_id: str, images: List[Image.Image]) -> int:
        """Save variation images"""
        session_dir = self.base_dir / training_id
        variations_dir = session_dir / "variations"
        
        for i, image in enumerate(images):
            image_path = variations_dir / f"variation_{i:03d}.png"
            image.save(image_path)
            
        return len(images)
    
    def create_dataset_config(self, training_id: str, name_object: str, subject_type: str = 'object') -> str:
        """Create dataset config for training"""
        session_dir = self.base_dir / training_id
        variations_dir = session_dir / "variations"
        dataset_dir = session_dir / "dataset"
        
        # Create annotations
        annotations = []
        variation_files = list(variations_dir.glob("*.png"))
        
        # List of diverse backgrounds
        backgrounds = [
            "in a park",
            "at the beach",
            "in a classroom",
            "in a city street",
            "in a forest",
            "in a kitchen",
            "in a futuristic city",
            "in a library",
            "in a garden",
            "in a snowy landscape",
            "in a coffee shop",
            "in a studio",
            "in a mountain landscape",
            "in a night market",
            "in a science lab"
        ]
        # If number of variations > backgrounds then repeat backgrounds
        while len(backgrounds) < len(variation_files):
            backgrounds = backgrounds * 2
        backgrounds = backgrounds[:len(variation_files)]
        
        for idx, img_path in enumerate(variation_files):
            background = backgrounds[idx]
            if subject_type == 'background':
                # Prompt only background
                caption = f"A photo of {background}"
                # Only generate text-to-image annotation, NO generate edit annotation
                annotations.append({
                    "task_type": "text_to_image",
                    "instruction": caption,
                    "input_images": None,
                    "output_image": str(img_path.absolute())
                })
            else:
                # Default: object + background
                caption = f"A photo of {name_object} {background}"
                edit_instruction = f"Change the background to {background}"
                # text-to-image annotation
                annotations.append({
                    "task_type": "text_to_image",
                    "instruction": caption,
                    "input_images": None,
                    "output_image": str(img_path.absolute())
                })
                # edit annotation (minh họa: input = output = variation image, instruction là edit background)
                annotations.append({
                    "task_type": "edit",
                    "instruction": edit_instruction,
                    "input_images": [str(img_path.absolute())],
                    "output_image": str(img_path.absolute())
                })
        
        # Save annotations as JSONL files (format mà OmniGen2 expects)
        jsonl_dir = dataset_dir / "jsonls"
        jsonl_dir.mkdir(exist_ok=True)
        
        # Split annotations by task type
        t2i_annotations = []
        edit_annotations = []
        
        for annotation in annotations:
            if annotation["task_type"] == "text_to_image":
                t2i_annotations.append({
                    "task_type": "t2i",
                    "instruction": annotation["instruction"],
                    "output_image": annotation["output_image"]
                })
            elif annotation["task_type"] == "edit":
                edit_annotations.append({
                    "task_type": "edit",
                    "instruction": annotation["instruction"],
                    "input_images": annotation["input_images"],
                    "output_image": annotation["output_image"]
                })
        
        # Save T2I annotations
        if t2i_annotations:
            t2i_path = jsonl_dir / "t2i.jsonl"
            with open(t2i_path, "w") as f:
                for annotation in t2i_annotations:
                    f.write(json.dumps(annotation) + "\n")
            logger.info(f"Saved {len(t2i_annotations)} T2I annotations to {t2i_path}")
        
        # Save Edit annotations
        if edit_annotations:
            edit_path = jsonl_dir / "edit.jsonl"
            with open(edit_path, "w") as f:
                for annotation in edit_annotations:
                    f.write(json.dumps(annotation) + "\n")
            logger.info(f"Saved {len(edit_annotations)} Edit annotations to {edit_path}")
        
        # Create dataset config in OmniGen2 format
        dataset_config = {
            "data": []
        }
        
        if t2i_annotations:
            dataset_config["data"].append({
                "path": str(t2i_path.absolute()),
                "type": "t2i",
                "ratio": 1.0
            })
        
        if edit_annotations:
            dataset_config["data"].append({
                "path": str(edit_path.absolute()),
                "type": "edit",
                "ratio": 1.0
            })
        
        # Log dataset config for debugging
        logger.info(f"Created dataset config: {dataset_config}")
        
        config_path = dataset_dir / "train_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        return str(config_path)
    
    def create_training_config(self, training_id: str, name_object: str, ultra_low_memory: bool = False, use_multi_gpu: bool = False, total_gpus: int = 1) -> str:
        """Create config file for OmniGen2 training with optional ultra-low memory mode and multi-GPU support"""
        session_dir = self.base_dir / training_id
        dataset_config_path = session_dir / "dataset" / "train_config.yml"
        
        # Determine batch sizes based on memory mode and GPU configuration
        if ultra_low_memory:
            batch_size = 1
            if use_multi_gpu:
                # Multi-GPU ultra low memory: can use slightly larger gradient accumulation
                gradient_accumulation_steps = 2  # Conservative for multi-GPU ultra low memory
                global_batch_size = batch_size * gradient_accumulation_steps * total_gpus  # 1 * 2 * 2 = 4
            else:
                gradient_accumulation_steps = 1  # Reduced for single GPU ultra low memory
                global_batch_size = 1  # 1 * 1 * 1 = 1
            dataloader_num_workers = 0  # No workers to save memory
            max_output_pixels = 65536  # 256x256 images - ultra small to prevent triton OOM
            max_input_pixels = [65536, 32768, 16384, 8192]  # Ultra small inputs
            lora_rank = 2  # Extremely small LoRA rank
        else:
            batch_size = 1
            if use_multi_gpu:
                # Multi-GPU standard: can use larger batch sizes with FSDP memory distribution
                gradient_accumulation_steps = 2  # Smaller accumulation since we have more GPUs
                global_batch_size = batch_size * gradient_accumulation_steps * total_gpus  # 1 * 2 * 2 = 4
                dataloader_num_workers = 2  # More workers for multi-GPU
                max_output_pixels = 1048576  # 1024x1024 images - larger for multi-GPU
                max_input_pixels = [1048576, 524288, 262144, 131072]  # Larger inputs
                lora_rank = 16  # Larger LoRA rank for better quality
            else:
                gradient_accumulation_steps = 4  # Increased for single GPU to maintain effective batch size
                global_batch_size = 4  # 1 * 4 * 1 = 4 (maintains same effective batch size as multi-GPU)
                dataloader_num_workers = 1
                max_output_pixels = 524288  # 768x768 images
                max_input_pixels = [524288, 524288, 262144, 131072]
                lora_rank = 8
        
        # Base config
        training_config = {
            "name": f"train_{name_object}_{training_id[:8]}",
            "seed": 2233,
            "device_specific_seed": True,
            "workder_specific_seed": True,
            
            "data": {
                "data_path": str(dataset_config_path.absolute()),
                "use_chat_template": True,
                "maximum_text_tokens": 256 if ultra_low_memory else 888,  # Reduce text tokens in ultra low memory
                "prompt_dropout_prob": 0.0001,
                "ref_img_dropout_prob": 0.1,
                "max_output_pixels": max_output_pixels,  # Dynamic based on memory mode
                "max_input_pixels": max_input_pixels,  # Dynamic based on memory mode
                "max_side_length": 1024 if ultra_low_memory else 2048,  # Reduce max side length
            },
            
            "model": {
                "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                "pretrained_model_path": "OmniGen2/OmniGen2",
                
                "arch_opt": {
                    "patch_size": 2,
                    "in_channels": 16,
                    "hidden_size": 2520,
                    "num_layers": 32,
                    "num_refiner_layers": 2,
                    "num_attention_heads": 21,
                    "num_kv_heads": 7,
                    "multiple_of": 256,
                    "norm_eps": 1e-05,
                    "axes_dim_rope": [40, 40, 40],
                    "axes_lens": [10000, 10000, 10000],
                    "text_feat_dim": 2048,
                    "timestep_scale": 1000
                }
            },
            
            "transport": {
                "snr_type": "lognorm",
                "do_shift": True,
                "dynamic_time_shift": True
            },
            
            "train": {
                "global_batch_size": global_batch_size,  # Properly calculated for single GPU
                "batch_size": batch_size,  # Always 1 for memory efficiency
                "gradient_accumulation_steps": gradient_accumulation_steps,  # Adjusted for single GPU
                "max_train_steps": 200 if ultra_low_memory else 1000,  # Drastically reduce steps for ultra low memory
                "dataloader_num_workers": dataloader_num_workers,  # Dynamic workers based on memory
                
                "learning_rate": 1e-6 if ultra_low_memory else 8e-7,  # Slightly higher LR for fewer steps
                "scale_lr": False,
                "lr_scheduler": "timm_constant_with_warmup",
                "warmup_t": 50 if ultra_low_memory else 100,  # Reduce warmup steps
                "warmup_lr_init": 1e-18,
                "warmup_prefix": True,
                "t_in_epochs": False,
                
                "use_8bit_adam": True,  # Enable 8-bit Adam for memory savings
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_weight_decay": 0.01,
                "adam_epsilon": 1e-08,
                "max_grad_norm": 0.5 if ultra_low_memory else 1,  # Smaller grad norm for stability
                
                "gradient_checkpointing": True,
                "set_grads_to_none": True,
                "allow_tf32": False,
                "mixed_precision": "fp16" if ultra_low_memory else "bf16",  # FP16 for ultra low memory
                "ema_decay": 0.0,
                
                # Advanced memory optimizations
                "dataloader_pin_memory": False,  # Disable pin memory to save GPU memory
                "cpu_offload_optimizer_state": True,  # Offload optimizer state to CPU
                "cpu_offload_params": True,  # Offload parameters to CPU when not in use
                
                # LoRA settings - ultra aggressive for memory
                "lora_ft": True,
                "lora_rank": lora_rank,  # Dynamic LoRA rank
                "lora_dropout": 0,
                
                # Disable DeepSpeed
                "deepspeed": None,
                "fsdp": None
            },
            
            "val": {
                "validation_steps": 500 if ultra_low_memory else 200,  # Less frequent validation
                "train_visualization_steps": 100 if ultra_low_memory else 50
            },
            
            "logger": {
                "log_with": [],  # Disable wandb for API training
                "checkpointing_steps": 100 if ultra_low_memory else 200,  # More frequent checkpoints for ultra low memory
                "checkpoints_total_limit": 2 if ultra_low_memory else 3  # Keep fewer checkpoints
            }
        }
        
        # Add missing parameters at root level
        training_config["resume_from_checkpoint"] = None
        training_config["cache_dir"] = None
        
        config_path = session_dir / "training_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        # Log training config for debugging
        mode_info = f"Multi-GPU FSDP ({total_gpus} GPUs)" if use_multi_gpu else f"Single GPU"
        logger.info(f"Created training config for {mode_info}: batch_size={batch_size}, gradient_accumulation_steps={gradient_accumulation_steps}, global_batch_size={global_batch_size}, LoRA_rank={lora_rank}")
            
        return str(config_path)

    def create_accelerate_config(self, gpu_id: int, memory_mode: str, use_multi_gpu: bool = False, total_gpus: int = 1) -> str:
        """Create dynamic accelerate config based on selected GPU and memory mode"""
        if use_multi_gpu and total_gpus > 1:
            # Multi-GPU FSDP configuration
            config_content = {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "FSDP",
                "downcast_bf16": "no",
                "machine_rank": 0,
                "main_process_ip": "localhost",
                "main_process_port": 29500,
                "main_training_function": "main",
                "mixed_precision": "fp16" if memory_mode == "extreme" else "bf16",
                "num_machines": 1,
                "num_processes": total_gpus,  # Use actual number of GPUs
                "rdzv_backend": "static",
                "same_network": True,
                "tpu_env": [],
                "tpu_use_cluster": False,
                "tpu_use_sudo": False,
                "use_cpu": False,
                # FSDP specific settings
                "fsdp_config": {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_backward_prefetch": "BACKWARD_PRE",
                    "fsdp_cpu_ram_efficient_loading": True if memory_mode in ["extreme", "ultra"] else False,
                    "fsdp_forward_prefetch": False,
                    "fsdp_offload_params": True if memory_mode == "extreme" else False,
                    "fsdp_sharding_strategy": "FULL_SHARD",  # Most memory efficient
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_sync_module_states": True,
                    "fsdp_transformer_layer_cls_to_wrap": "OmniGen2TransformerBlock",
                    "fsdp_use_orig_params": True
                }
            }
            config_path = f"accelerate_config_multigpu_{total_gpus}gpu_{memory_mode}.yml"
        else:
            # Single GPU configuration (existing code)
            config_content = {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "NO",
                "downcast_bf16": "no",
                "gpu_ids": str(gpu_id),
                "machine_rank": 0,
                "main_process_ip": "localhost",
                "main_process_port": 29500,
                "main_training_function": "main",
                "mixed_precision": "fp16" if memory_mode == "extreme" else "bf16",
                "num_machines": 1,
                "num_processes": 1,
                "rdzv_backend": "static",
                "same_network": True,
                "tpu_env": [],
                "tpu_use_cluster": False,
                "tpu_use_sudo": False,
                "use_cpu": False
            }
            config_path = f"accelerate_config_gpu{gpu_id}_{memory_mode}.yml"
        
        # Create dynamic config file
        with open(config_path, "w") as f:
            yaml.dump(config_content, f, default_flow_style=False)
        
        return config_path

    def get_gpu_status(self) -> Dict[str, Any]:
        """Get current GPU memory status for all available GPUs"""
        gpu_status = {
            "gpu_count": 0,
            "gpus": [],
            "recommended_gpu": None
        }
        
        if not torch.cuda.is_available():
            return gpu_status
            
        gpu_status["gpu_count"] = torch.cuda.device_count()
        gpu_memory_available = []
        
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_free = memory_total - memory_allocated
            gpu_memory_available.append(memory_free)
            
            gpu_info = {
                "gpu_id": i,
                "name": torch.cuda.get_device_properties(i).name,
                "memory_total_gb": round(memory_total, 2),
                "memory_allocated_gb": round(memory_allocated, 2),
                "memory_free_gb": round(memory_free, 2),
                "memory_utilization_percent": round((memory_allocated / memory_total) * 100, 1)
            }
            
            gpu_status["gpus"].append(gpu_info)
        
        # Recommend best GPU and multi-GPU feasibility
        if gpu_memory_available:
            best_gpu_id = gpu_memory_available.index(max(gpu_memory_available))
            gpu_status["recommended_gpu"] = best_gpu_id
            
            # Check multi-GPU feasibility
            total_memory = sum(gpu_memory_available)
            min_memory_per_gpu = min(gpu_memory_available)
            gpu_status["multi_gpu_feasible"] = (
                len(gpu_memory_available) >= 2 and 
                min_memory_per_gpu >= 10.0 and 
                total_memory >= 20.0
            )
            gpu_status["total_memory_gb"] = round(total_memory, 2)
            gpu_status["min_memory_per_gpu_gb"] = round(min_memory_per_gpu, 2)
            
        return gpu_status

    async def run_training(self, training_id: str, name_object: str, preferred_gpu_id: Optional[int] = None) -> bool:
        """Run training process using accelerate with adaptive memory optimization and smart GPU selection"""
        try:
            session_dir = self.base_dir / training_id
            
            # Force GPU memory cleanup before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
            
            # Check available GPU memory
            gpu_memory_available = []
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_free = memory_total - memory_allocated
                gpu_memory_available.append(memory_free)
                logger.info(f"GPU {i}: {memory_free:.2f}GB free out of {memory_total:.2f}GB")
            
            # Determine multi-GPU vs single-GPU strategy
            total_gpus = len(gpu_memory_available)
            total_memory = sum(gpu_memory_available)
            min_memory_per_gpu = min(gpu_memory_available) if gpu_memory_available else 0
            
            # Multi-GPU criteria: >= 2 GPUs, each with >= 10GB, and total >= 20GB
            use_multi_gpu = (
                total_gpus >= 2 and 
                min_memory_per_gpu >= 10.0 and 
                total_memory >= 20.0 and
                preferred_gpu_id is None  # Don't use multi-GPU if user specified single GPU
            )
            
            if use_multi_gpu:
                logger.info(f"🚀 Multi-GPU training enabled: {total_gpus} GPUs with {total_memory:.2f}GB total memory")
                logger.info(f"   GPU memory distribution: {[f'GPU{i}: {mem:.1f}GB' for i, mem in enumerate(gpu_memory_available)]}")
                best_gpu_memory = min_memory_per_gpu  # Use minimum for safety calculations
            else:
                # Single GPU selection
                if preferred_gpu_id is not None and 0 <= preferred_gpu_id < len(gpu_memory_available):
                    best_gpu_id = preferred_gpu_id
                    best_gpu_memory = gpu_memory_available[best_gpu_id]
                    logger.info(f"Using user-specified GPU {best_gpu_id} with {best_gpu_memory:.2f}GB free memory for training")
                else:
                    best_gpu_id = gpu_memory_available.index(max(gpu_memory_available))
                    best_gpu_memory = gpu_memory_available[best_gpu_id]
                    logger.info(f"Auto-selected GPU {best_gpu_id} with {best_gpu_memory:.2f}GB free memory for training")
            
            # Progressive memory optimization strategy - SMART GPU SELECTION WITH MULTI-GPU SUPPORT
            if best_gpu_memory < 8.0:  # Extremely limited memory - use extreme low memory mode
                if use_multi_gpu:
                    logger.error(f"Extremely limited GPU memory detected across {total_gpus} GPUs. Using FSDP extreme low memory mode with maximum CPU offloading.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training in FSDP EXTREME low memory mode on {total_gpus} GPUs...")
                    accelerate_config = self.create_accelerate_config(0, "extreme", use_multi_gpu=True, total_gpus=total_gpus)
                else:
                    logger.error(f"Extremely limited GPU memory detected on GPU {best_gpu_id}. Using extreme low memory mode with aggressive CPU offloading and Triton disabled.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training in EXTREME low memory mode on GPU {best_gpu_id}...")
                    accelerate_config = self.create_accelerate_config(best_gpu_id, "extreme")
                
                # Create ultra-low memory training config with smallest possible settings
                config_path = self.create_training_config(training_id, name_object, ultra_low_memory=True, use_multi_gpu=use_multi_gpu, total_gpus=total_gpus)
                
                # Extreme low memory training - maximum CPU offloading
                cmd = [
                    "accelerate", "launch",
                    "--config_file", accelerate_config,
                    "../OmniGen2/train.py",
                    "--config", config_path,
                ]
            elif best_gpu_memory < 10.0:  # Very limited memory - use ultra low memory mode
                if use_multi_gpu:
                    logger.warning(f"Limited GPU memory detected across {total_gpus} GPUs. Using FSDP ultra-low memory mode with CPU offloading.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training in FSDP ultra-low memory mode on {total_gpus} GPUs...")
                    accelerate_config = self.create_accelerate_config(0, "ultra", use_multi_gpu=True, total_gpus=total_gpus)
                else:
                    logger.warning(f"Very limited GPU memory detected on GPU {best_gpu_id}. Using ultra-low memory mode with maximum CPU offloading.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training in ultra-low memory mode on GPU {best_gpu_id}...")
                    accelerate_config = self.create_accelerate_config(best_gpu_id, "ultra")
                
                # Create ultra-low memory training config
                config_path = self.create_training_config(training_id, name_object, ultra_low_memory=True, use_multi_gpu=use_multi_gpu, total_gpus=total_gpus)
                
                # Ultra low memory training - maximum CPU offloading
                cmd = [
                    "accelerate", "launch",
                    "--config_file", accelerate_config,
                    "../OmniGen2/train.py",
                    "--config", config_path,
                ]
            else:  # Standard mode - single or multi-GPU based on criteria
                if use_multi_gpu:
                    logger.info(f"🚀 Using FSDP multi-GPU training on {total_gpus} GPUs with {total_memory:.2f}GB total memory for maximum performance.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training with FSDP on {total_gpus} GPUs (high performance mode)...")
                    accelerate_config = self.create_accelerate_config(0, "standard", use_multi_gpu=True, total_gpus=total_gpus)
                else:
                    logger.info(f"Using single GPU training on GPU {best_gpu_id} with {best_gpu_memory:.2f}GB memory for OmniGen2 stability.")
                    self.update_status(training_id, "training", 0.5, f"Starting OmniGen2 LoRA training on GPU {best_gpu_id} (stable mode)...")
                    accelerate_config = self.create_accelerate_config(best_gpu_id, "standard")
                
                # Create standard training config
                config_path = self.create_training_config(training_id, name_object, ultra_low_memory=False, use_multi_gpu=use_multi_gpu, total_gpus=total_gpus)
                
                # Standard training with optimizations
                cmd = [
                    "accelerate", "launch",
                    "--config_file", accelerate_config,
                    "../OmniGen2/train.py",
                    "--config", config_path,
                ]
            
            logger.info(f"Executing training command: {' '.join(cmd)}")
            
            # Set environment variables for aggressive memory management and CPU offloading
            env = os.environ.copy()
            
            # Base memory optimization settings
            base_env = {
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:64",  # Smaller split size
                "CUDA_LAUNCH_BLOCKING": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                # CPU offloading optimization
                "PYTORCH_CPU_OFFLOAD": "1",
                "FSDP_CPU_OFFLOAD": "1",
                # Memory cleanup
                "PYTORCH_CUDA_ALLOC_SYNC": "1",
                "CUDA_CACHE_DISABLE": "1",
                # CRITICAL: Disable Triton autotuning to prevent OOM in layer norm backward pass
                "TRITON_CACHE_DIR": "",  # Disable cache to force no autotuning
                "TRITON_DISABLE_AUTO_TUNE": "1",  # Disable autotuning completely
                "TRITON_AUTO_TUNE": "0",  # Alternative flag to disable autotuning
                "TRITON_AUTOTUNE": "0",  # Yet another flag
                # Force deterministic kernel selection without benchmarking
                "TRITON_FORCE_KERNEL": "1",
                "TRITON_SKIP_BENCHMARKS": "1",
            }
            
            # Memory-specific optimizations
            if best_gpu_memory < 8.0:  # Extreme low memory
                extreme_low_memory_env = {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:16,caching_allocator_allocation:0.5",
                    "PYTORCH_CUDA_MEMORY_FRACTION": "0.5",  # Very conservative GPU memory usage
                    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
                    "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log",
                    "TORCH_NCCL_BLOCKING_WAIT": "1",
                    "NCCL_ASYNC_ERROR_HANDLING": "1",
                    "PYTORCH_JIT": "0",  # Disable JIT compilation to save memory
                    "TORCH_COMPILE": "0",  # Disable torch.compile
                    "CUDA_MODULE_LOADING": "LAZY",  # Lazy module loading
                    # EXTREME: Force disable all Triton optimizations that could allocate memory
                    "TRITON_CACHE_MANAGER": "0",
                    "TRITON_ENABLE_PERF_MODEL": "0",
                    "TRITON_USE_ASSERT": "0",
                }
                # Only set CUDA_VISIBLE_DEVICES for single GPU mode
                if not use_multi_gpu:
                    extreme_low_memory_env["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
                base_env.update(extreme_low_memory_env)
            elif best_gpu_memory < 10.0:  # Ultra low memory
                ultra_low_memory_env = {
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:32,caching_allocator_allocation:0.8",
                    "PYTORCH_CUDA_MEMORY_FRACTION": "0.8",  # Limit GPU memory usage
                    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",  # Enable MPS for better memory sharing
                    "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-log",
                    "TORCH_NCCL_BLOCKING_WAIT": "1",
                    "NCCL_ASYNC_ERROR_HANDLING": "1",
                }
                # Only set CUDA_VISIBLE_DEVICES for single GPU mode
                if not use_multi_gpu:
                    ultra_low_memory_env["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
                base_env.update(ultra_low_memory_env)
            else:  # Standard mode
                if use_multi_gpu:
                    # Multi-GPU mode: let FSDP see all GPUs
                    standard_env = {
                        "PYTORCH_CUDA_MEMORY_FRACTION": "0.95",  # Use most of GPU memory
                        "NCCL_SOCKET_IFNAME": "lo",  # Use loopback for single-machine multi-GPU
                        "NCCL_P2P_DISABLE": "1",  # Disable P2P for stability
                        "NCCL_IB_DISABLE": "1",  # Disable InfiniBand
                    }
                else:
                    # Single GPU mode
                    standard_env = {
                        "CUDA_VISIBLE_DEVICES": str(best_gpu_id),  # Use GPU with most free memory
                        "PYTORCH_CUDA_MEMORY_FRACTION": "0.95",  # Use most of GPU memory
                    }
                base_env.update(standard_env)
            
            env.update(base_env)
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.cwd()), # Chạy từ thư mục gốc của project
                env=env,
            )
            
            # Monitor training progress with real-time output
            logger.info("Starting training process...")
            
            # Read output in real-time
            while True:
                output = await process.stdout.readline()
                if output:
                    line = output.decode().strip()
                    logger.info(f"Training: {line}")
                    
                    # Check for training progress
                    if "step" in line.lower() and "loss" in line.lower():
                        self.update_status(training_id, "training", 0.6, f"Training in progress: {line}")
                else:
                    break
            
            # Wait for process to complete
            return_code = await process.wait()
            
            # Get any remaining stderr
            stderr_data = await process.stderr.read()
            if stderr_data:
                stderr = stderr_data.decode()
                logger.warning(f"Training stderr: {stderr}")
            
            if return_code == 0:
                self.update_status(training_id, "converting", 0.8, "Converting model to HuggingFace format...")
                
                # Convert checkpoint to HF format
                await self.convert_checkpoint(training_id, config_path)
                
                return True
            else:
                error_message = stderr if stderr else "Unknown error"
                logger.error(f"Training failed with return code {return_code}:\n{error_message}")
                self.update_status(training_id, "failed", 0.0, f"Training failed: {error_message}")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.update_status(training_id, "failed", 0.0, f"Training error: {str(e)}")
            return False
    
    # ... (hàm convert_checkpoint và generate_test_image không thay đổi)
    async def convert_checkpoint(self, training_id: str, config_path: str):
        """Convert checkpoint to HuggingFace format"""
        try:
            session_dir = self.base_dir / training_id
            
            # Find latest checkpoint
            experiments_dir = Path("../OmniGen2/experiments")
            
            # Create experiments directory if it doesn't exist
            experiments_dir.mkdir(exist_ok=True)
            
            checkpoint_dirs = list(experiments_dir.glob("train_*_*/checkpoint-*"))
            
            if not checkpoint_dirs:
                raise Exception("No checkpoint found")
                
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
            model_path = latest_checkpoint / "pytorch_model.bin"
            save_path = session_dir / "model"
            
            cmd = [
                "python", "../OmniGen2/convert_ckpt_to_hf_format.py",
                "--config_path", config_path,
                "--model_path", str(model_path),
                "--save_path", str(save_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Conversion failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"Conversion error: {str(e)}")
            raise
    
    async def generate_test_image(self, training_id: str, name_object: str, omnigen2_model=None) -> Optional[str]:
        """Generate test image with trained model"""
        try:
            session_dir = self.base_dir / training_id
            model_path = session_dir / "model"
            output_dir = session_dir / "output"
            
            if not model_path.exists():
                raise Exception("Trained model not found")
            
            self.update_status(training_id, "generating", 0.9, "Generating test image...")
            
            
            # Use provided model or create new if not available
            if omnigen2_model is None:
                from models.omnigen2_model import OmniGen2Model
                model = OmniGen2Model()
            else:
                model = omnigen2_model
                
            # Load LoRA weights
            model.pipeline.load_lora_weights(str(model_path))
            
            # Generate test image
            prompt = f"A professional photo of {name_object}, high quality, detailed"
            result = model.generate_image(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=4.0,
                width=1024,
                height=1024
            )
            
            # Save image
            output_path = output_dir / "test_output.png"
            result["image"].save(output_path)
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Test generation error: {str(e)}")
            return None

# Global instance
training_manager = TrainingManager()