# Đồ án: Ứng dụng Sinh Ảnh AI Tùy Biến Đối Tượng & Con Người

## 1. Giới thiệu

Các ứng dụng Stable Diffusion hiện tại không thể sinh ra ảnh có chứa vật thể hoặc người mà model AI chưa từng train, dẫn đến hạn chế lớn về khả năng cá nhân hóa và sáng tạo. Project này xây dựng một hệ thống Image Generation mới, cho phép:
- Sinh ảnh chứa object hoặc người từ ảnh upload (không cần model đã train object đó).
- Tùy biến chi tiết (ví dụ: "ca sĩ A cầm trên tay cuốn sách thiết kế phần mềm").
- Kết hợp nhiều người/vật thể trên cùng một ảnh – điều mà các ứng dụng Stable Diffusion hiện tại chưa làm tốt.

## 2. Kiến trúc hệ thống

Hệ thống gồm 3 thành phần chính:
- **Backend (FastAPI)**: Xử lý AI inference, training, cung cấp API cho frontend.
- **Frontend (React)**: Giao diện người dùng, upload ảnh, nhập prompt, xem kết quả.
- **Model AI (DreamO, OmniGen2, DFloat11)**: Các mô hình sinh ảnh, chỉnh sửa ảnh, training LoRA.

### Sơ đồ tổng quan
```
Người dùng <-> Frontend (React) <-> Backend (FastAPI) <-> Model AI (DreamO, OmniGen2)
```

## 3. Tính năng nổi bật

### 3.1. DreamO
- Sinh ảnh từ nhiều reference images (tối đa 10), hỗ trợ các task: IP (Image Prompting), ID (Face Identity), Style (Style Transfer).
- Tùy biến object, human, style, thử đồ ảo, multi-condition.

### 3.2. OmniGen2
- Kết hợp nhiều người/vật thể từ nhiều ảnh khác nhau lên cùng một ảnh mới (in-context generation).
- Chỉnh sửa ảnh theo hướng dẫn text (image editing).
- Hỗ trợ nén model (DFloat11) để chạy trên GPU 16GB.

### 3.3. Training Pipeline
- Tự động hóa training LoRA cho object mới từ ảnh upload.
- Quản lý, theo dõi, inference với model đã train.

## 4. Hướng dẫn cài đặt & chạy

### 4.1. Backend (FastAPI)

#### Yêu cầu:
- Python 3.11
- 1 hoặc 2 GPU (tối thiểu 16GB, khuyến nghị 24GB VRAM)
- CUDA 11.8+ hoặc 12.1+

#### Cài đặt:
```bash
# Tạo môi trường
conda create -n ai-backend python=3.11
conda activate ai-backend
cd backend

# Cài đặt PyTorch (chọn đúng CUDA)
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Cài các thư viện khác
pip install -r requirements.txt

# (Khuyến nghị) Cài Nunchaku để tối ưu VRAM
# Xem: https://github.com/mit-han-lab/nunchaku

# Cài FaceXLib cho DreamO
pip install git+https://github.com/ToTheBeginning/facexlib.git

# Cài dependencies cho training
pip install accelerate transformers diffusers timm PyYAML datasets huggingface_hub
python setup_accelerate.py
# hoặc
bash install_dependencies.sh
```

#### Chạy backend:
```bash
# Test model
python test_models.py

# Khởi động backend
python start_backend.py
# hoặc
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
- API docs: http://localhost:8000/docs

### 4.2. Frontend (React)

#### Yêu cầu:
- Node.js 18+

#### Cài đặt & chạy:
```bash
cd frontend-2
npm install
npm run dev
```
- Truy cập: http://localhost:3000

> Lưu ý: Backend phải chạy trước ở http://localhost:8000

### 4.3. Model AI (DreamO, OmniGen2, DFloat11)

- Đã được tích hợp sẵn trong backend, chỉ cần cài đúng dependencies và tải model về (sẽ tự động khi chạy lần đầu).
- Có thể chạy demo riêng lẻ:
  - DreamO: `python DreamO/app.py`
  - OmniGen2: `cd OmniGen2-DFloat11 && python app.py`

## 5. Hướng dẫn sử dụng

### 5.1. Sinh ảnh với DreamO
- Chọn tab DreamO trên frontend.
- Nhập prompt mô tả ảnh.
- Upload 1-10 ảnh reference (chọn task: IP/ID/Style cho từng ảnh).
- Điều chỉnh tham số nếu cần (width, height, steps, guidance...).
- Nhấn "Tạo ảnh".

### 5.2. Kết hợp nhiều người/vật thể (OmniGen2)
- Chọn tab OmniGen2 → In-Context Generation.
- Nhập instruction (ví dụ: "Let the person in image 2 hold the toy from image 1").
- Upload 1-5 ảnh input.
- Nhấn "Tạo Composition".

### 5.3. Training object mới
- Chọn tab Training → New Training.
- Nhập tên object, upload 1-5 ảnh reference.
- Nhấn "Bắt đầu Training" và theo dõi tiến trình.

## 6. Một số lưu ý & troubleshooting

- Đảm bảo backend và frontend chạy đúng port.
- Kiểm tra GPU đủ VRAM, cài đúng CUDA.
- Nếu lỗi CORS, kiểm tra cấu hình backend.
- Ảnh upload: JPEG/PNG/WebP, <10MB.
- Nếu chậm, giảm resolution hoặc số bước inference.

## 7. Đóng góp & liên hệ

- Dự án sử dụng mã nguồn mở, khuyến khích đóng góp.
- Mọi thắc mắc vui lòng liên hệ qua email hoặc GitHub Issues.

---

**Chúc bạn thành công với đồ án AI Image Generation!** 