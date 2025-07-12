# AI Image Generation Backend

Backend FastAPI cho DreamO và OmniGen2 image generation services với cấu hình GPU tối ưu.

## 🚀 Tính năng

### DreamO API
- **Multi-reference generation**: Hỗ trợ tối đa 10 ảnh reference với các task khác nhau
- **Task types**: IP (Image Prompting), ID (Face Identity), Style (Style Transfer)
- **Nunchaku optimization**: Giảm VRAM xuống 6.5GB với tốc độ nhanh 2-4x
- **Debug images**: Xem kết quả preprocessing (background removal, face crop)
- **GPU**: Chạy trên cuda:0

### OmniGen2 API
- **In-context generation**: Tổng hợp nhiều object từ các ảnh khác nhau
- **Image editing**: Chỉnh sửa ảnh theo hướng dẫn text
- **DFloat11 compression**: Model nén 32% mà không mất chất lượng
- **Multiple schedulers**: Euler và DPMSolver
- **GPU**: Chạy trên cuda:1 (fallback về cuda:0 nếu chỉ có 1 GPU)

## 🖥️ GPU Configuration

### Yêu cầu hệ thống
- **Minimum**: 1 GPU với 24GB VRAM
- **Recommended**: 2 GPUs với 24GB VRAM mỗi GPU
- **CUDA**: 11.8+ hoặc 12.1+

### Cấu hình GPU
```
GPU 0 (cuda:0): DreamO model
GPU 1 (cuda:1): OmniGen2 model
```

Nếu chỉ có 1 GPU, cả hai models sẽ chạy trên cuda:0.

## 📁 Cấu trúc

```
backend/
├── main.py                 # FastAPI app chính
├── models/                 # Model wrappers
│   ├── dreamo_model.py     # DreamO wrapper (cuda:0)
│   └── omnigen2_model.py   # OmniGen2 wrapper (cuda:1)
├── routes/                 # API endpoints
│   ├── dreamo.py           # DreamO routes
│   └── omnigen2.py         # OmniGen2 routes
├── schemas/                # Pydantic schemas
│   ├── dreamo_schemas.py   # DreamO validation
│   └── omnigen2_schemas.py # OmniGen2 validation
├── utils/                  # Utility functions
│   └── image_utils.py      # Image processing
├── test_models.py          # Test script cho models
├── start_backend.py        # Script khởi động backend
└── requirements.txt        # Dependencies
```

## 🛠️ Cài đặt

### 1. Chuẩn bị môi trường

```bash
# Tạo conda environment
conda create -n ai-backend python=3.11
conda activate ai-backend

# Di chuyển đến thư mục backend
cd backend
```

### 2. Cài đặt dependencies

```bash
# Cài đặt PyTorch với CUDA (chọn version phù hợp)
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Cài đặt các dependencies khác
pip install -r requirements.txt

# Cài đặt Nunchaku (optional nhưng được khuyến nghị)
# Xem hướng dẫn tại: https://github.com/mit-han-lab/nunchaku
```

### 3. Cài đặt FaceXLib cho DreamO

```bash
pip install git+https://github.com/ToTheBeginning/facexlib.git
```

### 4. Cài đặt Dependencies cho Training

```bash
# Cài đặt dependencies cần thiết
pip install accelerate transformers diffusers timm PyYAML datasets huggingface_hub

# Cấu hình accelerate để tránh DeepSpeed
python setup_accelerate.py

# Hoặc chạy script tự động
bash install_dependencies.sh

# Kiểm tra accelerate config
python check_accelerate.py
```

**Lưu ý**: Training pipeline cần accelerate được cấu hình đúng để tránh lỗi DeepSpeed.

## 🚀 Chạy Backend

### Kiểm tra môi trường trước

```bash
# Test models loading
python test_models.py

# Kiểm tra GPU configuration
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### Development Mode

```bash
# Chạy với script tự động
python start_backend.py

# Hoặc chạy trực tiếp
python main.py

# Hoặc với uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
# Sử dụng script production
python start_backend.py

# Hoặc với uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Chỉ sử dụng 1 worker vì models rất nặng và không thể share giữa các processes.

### Kiểm tra trạng thái

```bash
# Health check tổng quát
curl http://localhost:8000/health

# Kiểm tra từng model
curl http://localhost:8000/api/dreamo/health
curl http://localhost:8000/api/omnigen2/health
```

## 📚 API Documentation

Sau khi chạy backend, truy cập:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

## 🎯 Sử dụng APIs

### DreamO API

#### Generate Image
```bash
curl -X POST "http://localhost:8000/api/dreamo/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a person playing guitar in the street",
    "ref_images": [
      {
        "image_data": "data:image/jpeg;base64,...",
        "task": "ip"
      }
    ],
    "width": 1024,
    "height": 1024,
    "num_steps": 12,
    "guidance": 4.5
  }'
```

#### Multi-reference Example
```json
{
  "prompt": "A girl wearing a shirt and skirt on the beach",
  "ref_images": [
    {
      "image_data": "data:image/jpeg;base64,...",
      "task": "id"
    },
    {
      "image_data": "data:image/jpeg;base64,...", 
      "task": "ip"
    },
    {
      "image_data": "data:image/jpeg;base64,...",
      "task": "ip"
    }
  ]
}
```

### OmniGen2 API

#### In-context Generation
```bash
curl -X POST "http://localhost:8000/api/omnigen2/in-context-generation" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Let the person in image 2 hold the toy from image 1",
    "input_images": [
      "data:image/jpeg;base64,...",
      "data:image/jpeg;base64,..."
    ],
    "text_guidance_scale": 5.0,
    "image_guidance_scale": 2.0
  }'
```

#### Image Editing
```bash
curl -X POST "http://localhost:8000/api/omnigen2/edit" \
  -H "Content-Type: application/json" \
  -d '{
    "instruction": "Change the background to classroom",
    "input_image": "data:image/jpeg;base64,...",
    "text_guidance_scale": 5.0,
    "image_guidance_scale": 2.0
  }'
```

## ⚙️ Cấu hình

### DreamO Settings
- **Task types**: `ip`, `id`, `style`
- **Max references**: 10 images
- **Recommended steps**: 12 (với turbo)
- **Memory optimization**: Nunchaku + offload

### OmniGen2 Settings
- **Max input images**: 5
- **Schedulers**: `euler`, `dpmsolver`
- **Recommended steps**: 50
- **Memory optimization**: DFloat11 compression

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
# Cho DreamO - sử dụng nunchaku
# Model sẽ tự động được cấu hình với nunchaku + offload

# Cho OmniGen2 - enable offloading
# Có thể modify trong omnigen2_model.py:
# self.config["enable_sequential_cpu_offload"] = True
```

### Model Loading Issues
```bash
# Kiểm tra paths
ls -la ../DreamO/
ls -la ../OmniGen2-DFloat11/

# Kiểm tra dependencies
pip list | grep torch
pip list | grep transformers
```

### Import Errors
```bash
# Đảm bảo PYTHONPATH đúng
export PYTHONPATH="${PYTHONPATH}:../DreamO:../OmniGen2-DFloat11"
```

### Training Pipeline Errors

#### Lỗi 'dict' object has no attribute 'images'
```bash
# Đã được sửa trong phiên bản mới
# Kiểm tra training pipeline:
python test_training.py
```

#### Lỗi Model không tương thích
```bash
# Test từng component:
python test_models.py
python test_training.py
```

#### Lỗi Memory trong Training
```bash
# Giảm batch size trong training_config.yml:
# "global_batch_size": 4,  # Thay vì 8
# "batch_size": 1,
# "gradient_accumulation_steps": 4,  # Thay vì 8
```

#### Lỗi DeepSpeed not installed
```bash
# Cài đặt dependencies và cấu hình accelerate:
pip install accelerate transformers diffusers timm PyYAML datasets huggingface_hub
python setup_accelerate.py

# Hoặc chạy script tự động:
bash install_dependencies.sh
```

#### Lỗi Tokenizers Parallelism Warning
```bash
# Đã được sửa trong code, nhưng có thể set environment variable:
export TOKENIZERS_PARALLELISM=false
```

#### Lỗi Accelerate Arguments
```bash
# Nếu gặp lỗi "unrecognized arguments", hãy kiểm tra accelerate config:
python check_accelerate.py

# Hoặc setup lại accelerate config:
python setup_accelerate.py
```

#### Lỗi File Not Found
```bash
# Kiểm tra tất cả đường dẫn cần thiết:
python check_paths.py

# Đảm bảo cấu trúc thư mục đúng:
# TKPM/
# ├── backend/
# ├── DreamO/
# └── OmniGen2-DFloat11/
```

## 📊 Performance

### DreamO (Nunchaku Mode)
- **RTX 3080**: ~20 giây (1024x1024)
- **RTX 4090**: ~15 giây (1024x1024)
- **Memory**: 6.5GB VRAM

### OmniGen2 (DFloat11)
- **A100**: ~25 giây (1024x1024)
- **RTX 4090**: ~30 giây (1024x1024)
- **Memory**: 14.3GB VRAM

## 🔍 Monitoring

### Health Checks
```bash
# Overall health
curl http://localhost:8000/health

# DreamO specific
curl http://localhost:8000/api/dreamo/health

# OmniGen2 specific  
curl http://localhost:8000/api/omnigen2/health
```

### Logs
```bash
# Xem logs realtime
tail -f /path/to/logs

# Check memory usage
nvidia-smi

# Monitor GPU utilization
watch -n 1 nvidia-smi
```

## 🌐 Frontend Integration

### JavaScript Example
```javascript
async function generateDreamO(prompt, refImages) {
  const response = await fetch('http://localhost:8000/api/dreamo/generate', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt: prompt,
      ref_images: refImages.map(img => ({
        image_data: img.data,
        task: img.task
      }))
    })
  });
  
  return await response.json();
}
```

### Python Client Example
```python
import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# DreamO
response = requests.post('http://localhost:8000/api/dreamo/generate', json={
    "prompt": "a person playing guitar",
    "ref_images": [{
        "image_data": f"data:image/jpeg;base64,{encode_image('person.jpg')}",
        "task": "ip"
    }]
})

# OmniGen2  
response = requests.post('http://localhost:8000/api/omnigen2/edit', json={
    "instruction": "Change background to beach",
    "input_image": f"data:image/jpeg;base64,{encode_image('input.jpg')}"
})
```

## 📋 Examples

Xem thêm examples tại:
- `GET /api/dreamo/examples`
- `GET /api/omnigen2/examples`

## 🔗 Links

- [DreamO GitHub](https://github.com/bytedance/DreamO)
- [OmniGen2 Paper](https://arxiv.org/abs/2506.18871)
- [DFloat11 GitHub](https://github.com/LeanModels/DFloat11)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🐛 Issues

Nếu gặp lỗi, vui lòng:
1. Kiểm tra logs
2. Verify model paths
3. Check CUDA/memory status
4. Create issue với thông tin chi tiết 