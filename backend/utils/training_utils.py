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
    """Manager class cho toàn bộ training pipeline"""
    
    def __init__(self, base_dir: str = "training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.training_status: Dict[str, Dict] = {}
        
    def create_training_session(self, name_object: str, description: Optional[str] = None) -> str:
        """Tạo session mới cho training"""
        training_id = str(uuid.uuid4())
        session_dir = self.base_dir / training_id
        session_dir.mkdir(exist_ok=True)
        
        # Tạo các thư mục con
        (session_dir / "variations").mkdir(exist_ok=True)
        (session_dir / "dataset").mkdir(exist_ok=True)
        (session_dir / "model").mkdir(exist_ok=True)
        (session_dir / "output").mkdir(exist_ok=True)
        
        # Lưu metadata
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
        """Cập nhật status của training session"""
        if training_id not in self.training_status:
            self.training_status[training_id] = {}
            
        self.training_status[training_id].update({
            "status": status,
            "progress": progress,
            "message": message
        })
        
        # Lưu vào file
        session_dir = self.base_dir / training_id
        if session_dir.exists():
            with open(session_dir / "status.json", "w") as f:
                json.dump(self.training_status[training_id], f, indent=2)
    
    def get_status(self, training_id: str) -> Dict:
        """Lấy status của training session"""
        return self.training_status.get(training_id, {
            "status": "not_found",
            "progress": 0.0,
            "message": "Training session not found"
        })
    
    def save_variations(self, training_id: str, images: List[Image.Image]) -> int:
        """Lưu các variation images"""
        session_dir = self.base_dir / training_id
        variations_dir = session_dir / "variations"
        
        for i, image in enumerate(images):
            image_path = variations_dir / f"variation_{i:03d}.png"
            image.save(image_path)
            
        return len(images)
    
    def create_dataset_config(self, training_id: str, name_object: str) -> str:
        """Tạo dataset config cho training"""
        session_dir = self.base_dir / training_id
        variations_dir = session_dir / "variations"
        dataset_dir = session_dir / "dataset"
        
        # Tạo annotations
        annotations = []
        variation_files = list(variations_dir.glob("*.png"))
        
        for img_path in variation_files:
            # Tạo diverse captions cho training
            captions = [
                f"A photo of {name_object}",
                f"An image of {name_object}",
                f"{name_object} in high quality",
                f"Professional photo of {name_object}",
                f"Clear image showing {name_object}",
            ]
            
            for i, caption in enumerate(captions):
                annotations.append({
                    "task_type": "text_to_image",
                    "instruction": caption,
                    "input_images": None,
                    "output_image": str(img_path.absolute())
                })
        
        # Lưu dataset config
        dataset_config = {
            "datasets": [{
                "type": "annotation_list",
                "data": annotations
            }]
        }
        
        config_path = dataset_dir / "train_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
            
        return str(config_path)
    
    def create_training_config(self, training_id: str, name_object: str) -> str:
        """Tạo config file cho OmniGen2 training"""
        session_dir = self.base_dir / training_id
        dataset_config_path = session_dir / "dataset" / "train_config.yml"
        
        # Base config template
        training_config = {
            "name": f"train_{name_object}_{training_id[:8]}",
            "seed": 2233,
            "device_specific_seed": True,
            "workder_specific_seed": True,
            
            "data": {
                "data_path": str(dataset_config_path.absolute()),
                "use_chat_template": True,
                "maximum_text_tokens": 888,
                "prompt_dropout_prob": 0.0001,
                "ref_img_dropout_prob": 0.1,
                "max_output_pixels": 1048576,
                "max_input_pixels": [1048576, 1048576, 589824, 262144],
                "max_side_length": 2048,
            },
            
            "model": {
                "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "pretrained_text_encoder_model_name_or_path": "Qwen/Qwen2.5-VL-3B-Instruct",
                "pretrained_model_path": "DFloat11/OmniGen2-transformer-DF11",
                
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
                "global_batch_size": 8,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "max_train_steps": 1000,  # Reduced for faster training
                "dataloader_num_workers": 2,
                
                "learning_rate": 1e-6,
                "scale_lr": False,
                "lr_scheduler": "timm_constant_with_warmup",
                "warmup_t": 100,
                "warmup_lr_init": 1e-18,
                "warmup_prefix": True,
                "t_in_epochs": False,
                
                "use_8bit_adam": False,
                "adam_beta1": 0.9,
                "adam_beta2": 0.95,
                "adam_weight_decay": 0.01,
                "adam_epsilon": 1e-08,
                "max_grad_norm": 1,
                
                "gradient_checkpointing": True,
                "set_grads_to_none": True,
                "allow_tf32": False,
                "mixed_precision": "bf16",
                "ema_decay": 0.0,
                
                # LoRA settings
                "lora_ft": True,
                "lora_rank": 16,
                "lora_dropout": 0.1,
                
                # Disable DeepSpeed
                "deepspeed": None,
                "fsdp": None
            },
            
            "val": {
                "validation_steps": 200,
                "train_visualization_steps": 50
            },
            
            "logger": {
                "log_with": [],  # Disable wandb for API training
                "checkpointing_steps": 200,
                "checkpoints_total_limit": 3
            }
        }
        
        config_path = session_dir / "training_config.yml"
        with open(config_path, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False)
            
        return str(config_path)

    async def run_training(self, training_id: str, name_object: str) -> bool:
        """Chạy training process"""
        try:
            session_dir = self.base_dir / training_id
            config_path = self.create_training_config(training_id, name_object)
            
            self.update_status(training_id, "training", 0.5, "Starting OmniGen2 LoRA training...")
            
            # Set environment variables to avoid tokenizers warning
            import os
            env = os.environ.copy()
            env["TOKENIZERS_PARALLELISM"] = "false"
            
            # Chạy training command không sử dụng DeepSpeed
            cmd = [
                "accelerate", "launch",
                "--num_processes", "1",
                "--mixed_precision", "bf16",
                "../OmniGen2-DFloat11/train.py",  # Sử dụng đường dẫn tương đối từ backend/
                "--config", config_path,
                "--global_batch_size", "8"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(Path.cwd()),
                env=env
            )
            
            # Monitor training progress
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.update_status(training_id, "converting", 0.8, "Converting model to HuggingFace format...")
                
                # Convert checkpoint to HF format
                await self.convert_checkpoint(training_id, config_path)
                
                return True
            else:
                logger.error(f"Training failed: {stderr.decode()}")
                self.update_status(training_id, "failed", 0.0, f"Training failed: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            self.update_status(training_id, "failed", 0.0, f"Training error: {str(e)}")
            return False
    
    async def convert_checkpoint(self, training_id: str, config_path: str):
        """Convert checkpoint sang HuggingFace format"""
        try:
            session_dir = self.base_dir / training_id
            
            # Tìm checkpoint mới nhất
            experiments_dir = Path("../OmniGen2-DFloat11/experiments")  # Đường dẫn từ backend/
            
            # Tạo thư mục experiments nếu chưa tồn tại
            experiments_dir.mkdir(exist_ok=True)
            
            checkpoint_dirs = list(experiments_dir.glob("train_*_*/checkpoint-*"))
            
            if not checkpoint_dirs:
                raise Exception("No checkpoint found")
                
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[1]))
            model_path = latest_checkpoint / "pytorch_model.bin"
            save_path = session_dir / "model"
            
            cmd = [
                "python", "../OmniGen2-DFloat11/convert_ckpt_to_hf_format.py",  # Đường dẫn từ backend/
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
        """Generate test image với trained model"""
        try:
            session_dir = self.base_dir / training_id
            model_path = session_dir / "model"
            output_dir = session_dir / "output"
            
            if not model_path.exists():
                raise Exception("Trained model not found")
            
            self.update_status(training_id, "generating", 0.9, "Generating test image...")
            
            # Sử dụng provided model hoặc tạo mới nếu không có
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