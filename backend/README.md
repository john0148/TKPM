# AI Image Generation Backend

Backend FastAPI cho DreamO và OmniGen2 image generation services.

## 🚀 Tính năng

### DreamO API
- **Multi-reference generation**: Hỗ trợ tối đa 10 ảnh reference với các task khác nhau
- **Task types**: IP (Image Prompting), ID (Face Identity), Style (Style Transfer)
- **Nunchaku optimization**: Giảm VRAM xuống 6.5GB với tốc độ nhanh 2-4x
- **Debug images**: Xem kết quả preprocessing (background removal, face crop)

### OmniGen2 API
- **In-context generation**: Tổng hợp nhiều object từ các ảnh khác nhau
- **Image editing**: Chỉnh sửa ảnh theo hướng dẫn text
- **DFloat11 compression**: Model nén 32% mà không mất chất lượng
- **Multiple schedulers**: Euler và DPMSolver

## 📁 Cấu trúc

```
backend/
├── main.py                 # FastAPI app chính
├── models/                 # Model wrappers
│   ├── dreamo_model.py     # DreamO wrapper
│   └── omnigen2_model.py   # OmniGen2 wrapper
├── routes/                 # API endpoints
│   ├── dreamo.py           # DreamO routes
│   └── omnigen2.py         # OmniGen2 routes
├── schemas/                # Pydantic schemas
│   ├── dreamo_schemas.py   # DreamO validation
│   └── omnigen2_schemas.py # OmniGen2 validation
├── utils/                  # Utility functions
│   └── image_utils.py      # Image processing
└── requirements.txt        # Dependencies
```

## 🛠️ Cài đặt

### 1. Chuẩn bị môi trường

```bash
# Tạo conda environment
conda create -n ai-backend python=3.10
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

## 🚀 Chạy Backend

### Development Mode

```bash
# Chạy với hot reload
python main.py

# Hoặc với uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

**Note**: Chỉ sử dụng 1 worker vì models rất nặng và không thể share giữa các processes.

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