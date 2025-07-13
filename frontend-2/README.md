# 🎨 AI Image Generation Frontend

Giao diện React đơn giản và đẹp mắt để sử dụng tất cả API từ backend AI Image Generation, bao gồm DreamO, OmniGen2 và Training pipeline.

## ✨ Tính năng

### 🎨 DreamO
- **Image Generation với Reference Images**: Tạo ảnh dựa trên nhiều reference images
- **Multi-task Support**: 
  - IP (Image Prompting) - General objects, characters, animals
  - ID (Identity Preservation) - Face identity preservation
  - Style (Style Transfer) - Artistic style application
- **Tối đa 10 reference images** với task type riêng biệt
- **Real-time preview** của reference images
- **Customizable parameters**: Width, Height, Steps, Guidance, Seed, etc.

### 🤖 OmniGen2
- **In-Context Generation**: Tạo composition từ nhiều input images
- **Image Editing**: Chỉnh sửa ảnh dựa trên text instructions
- **Advanced Parameters**: Text/Image guidance, scheduler selection
- **Multiple Schedulers**: Euler, DPM Solver
- **Tối đa 5 input images** cho in-context generation

### 🎓 Training
- **Custom Model Training**: Huấn luyện LoRA model cho objects riêng
- **Automated Pipeline**: 
  1. Generate 15 variations từ reference images (DreamO)
  2. Chuẩn bị dataset cho training
  3. Train LoRA model với OmniGen2  
  4. Generate test image
- **Training Management**: Xem, quản lý và xóa training sessions
- **Inference với Trained Models**: Sử dụng custom models đã train
- **Real-time Progress Tracking** với status updates

### 💊 Health Monitoring
- **System Health Check**: Kiểm tra trạng thái backend và models
- **Model Information**: Chi tiết về capabilities và requirements
- **Memory Usage Monitoring**: Theo dõi VRAM usage
- **Performance Metrics**: Inference time và system recommendations

## 🚀 Cài đặt và Chạy

### Prerequisites
- Node.js 18+ 
- npm hoặc yarn
- Backend AI Image Generation đang chạy tại `http://localhost:8000`

### Cài đặt Dependencies
```bash
npm install
```

### Chạy Development Server
```bash
npm run dev
```

Frontend sẽ chạy tại `http://localhost:3000`

### Build cho Production
```bash
npm run build
```

### Preview Production Build
```bash
npm run preview
```

## 🛠️ Cấu trúc Project

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── DreamOComponent.jsx      # DreamO interface
│   │   ├── OmniGen2Component.jsx    # OmniGen2 interface  
│   │   ├── TrainingComponent.jsx    # Training interface
│   │   └── HealthComponent.jsx      # Health monitoring
│   ├── utils/
│   │   └── api.js           # API utilities và helper functions
│   ├── App.jsx              # Main app component
│   ├── App.css              # Styles
│   └── main.jsx             # Entry point
├── index.html               # HTML template
├── vite.config.js           # Vite configuration
├── package.json             # Dependencies và scripts
└── README.md                # Tài liệu này
```

## 🎯 Sử dụng

### 1. DreamO - Image Generation
1. Chọn tab **DreamO**
2. Nhập **prompt** mô tả ảnh muốn tạo
3. Upload **1-10 reference images**
4. Chọn **task type** cho mỗi ảnh:
   - **IP**: General objects/characters
   - **ID**: Face identity preservation  
   - **Style**: Style transfer
5. Điều chỉnh parameters nếu cần
6. Nhấn **"Tạo ảnh"**

### 2. OmniGen2 - Advanced Generation & Editing

#### In-Context Generation:
1. Chọn tab **OmniGen2** → **In-Context Generation**
2. Nhập **instruction** tham chiếu đến images
   - VD: "Let the person in image 2 hold the toy from image 1 in a parking lot"
3. Upload **1-5 input images**
4. Điều chỉnh parameters
5. Nhấn **"Tạo Composition"**

#### Image Editing:
1. Chọn tab **OmniGen2** → **Image Editing**
2. Upload **ảnh cần chỉnh sửa**
3. Nhập **instruction** mô tả thay đổi
   - VD: "Change background to classroom", "Add a hat", "Remove the cat"
4. Nhấn **"Chỉnh sửa ảnh"**

### 3. Training - Custom Model Training

#### New Training:
1. Chọn tab **Training** → **New Training**
2. Nhập **tên object** (trigger word)
3. Upload **1-5 reference images** của object
4. Nhập **mô tả** (tùy chọn)
5. Nhấn **"Bắt đầu Training"**
6. Theo dõi **progress real-time**

#### Manage Sessions:
1. Chọn **Manage Sessions** để xem tất cả training sessions
2. Xóa sessions không cần thiết

#### Inference với Trained Model:
1. Chọn **Inference**
2. Chọn **trained model** từ dropdown
3. Nhập **prompt** sử dụng object đã train
4. Nhấn **"Tạo ảnh với Trained Model"**

### 4. Health - System Monitoring
1. Chọn tab **Health**
2. Xem trạng thái **General System**, **DreamO**, **OmniGen2**
3. Kiểm tra **memory usage** và **model information**
4. Nhấn **"Refresh Health Status"** để cập nhật

## 🔧 Configuration

### API Endpoint
Backend API endpoint được cấu hình trong `vite.config.js`:
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true
    }
  }
}
```

### Image Validation
Trong `src/utils/api.js`:
- **Supported formats**: JPEG, PNG, WebP
- **Max file size**: 10MB
- **Auto-resize**: Nếu ảnh quá lớn

## 🎨 UI/UX Features

- **Responsive Design**: Hoạt động tốt trên desktop và mobile
- **Modern UI**: Gradient backgrounds, smooth animations
- **Real-time Preview**: Xem ảnh upload ngay lập tức
- **Progress Tracking**: Visual progress bars cho training
- **Error Handling**: Thông báo lỗi rõ ràng
- **Loading States**: Spinner và loading messages
- **Status Badges**: Color-coded status indicators
- **Health Indicators**: Visual health status với colors

## 🔍 Troubleshooting

### Backend Connection Issues
- Đảm bảo backend đang chạy tại `http://localhost:8000`
- Kiểm tra CORS settings trong backend
- Xem browser console để debug API calls

### Image Upload Issues  
- Kiểm tra file format (JPEG/PNG/WebP only)
- Đảm bảo file < 10MB
- Thử resize ảnh nếu quá lớn

### Training Issues
- Cần ít nhất 1 reference image
- Tên object phải có ít nhất 2 ký tự
- Theo dõi training progress để xem lỗi

### Performance Issues
- Giảm image resolution nếu chậm
- Giảm number of inference steps
- Kiểm tra GPU memory usage

## 📝 Development Notes

- **React 18** với functional components và hooks
- **Vite** cho fast development và building
- **Pure CSS** không sử dụng external UI libraries
- **Modular Architecture** với reusable API utilities
- **Error Boundaries** và proper error handling
- **Memory Management** cho file uploads và previews

## 🚀 Future Enhancements

- [ ] Drag & drop cho image uploads
- [ ] Batch processing cho multiple images
- [ ] History/Gallery để lưu results
- [ ] Export/Download functionality
- [ ] User authentication và saved sessions
- [ ] Real-time notifications
- [ ] Advanced parameter presets
- [ ] Image comparison tools

## 📄 License

Cùng license với backend project.

---

**Enjoy creating amazing AI-generated images! 🎨✨** 