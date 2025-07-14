import React, { useState, useRef } from 'react';
import { omnigen2InContextGeneration, omnigen2EditImage, fileToBase64, validateImageFile } from '../utils/api';

const OmniGen2Component = ({ formData, onFormDataChange, onConfigOpen, onImageGenerated }) => {
  const [mode, setMode] = useState('in-context'); // 'in-context' or 'edit'
  const [inputImages, setInputImages] = useState([]);
  const [editImage, setEditImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const multiFileInputRef = useRef(null);
  const singleFileInputRef = useRef(null);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    onFormDataChange(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleModeChange = (newMode) => {
    setMode(newMode);
    setInputImages([]);
    setEditImage(null);
    setResult(null);
    setError(null);
    setIsDragOver(false);
  };

  const processMultiFiles = async (files) => {
    try {
      files.forEach(file => validateImageFile(file));
      
      if (files.length > 5) {
        throw new Error('Tối đa 5 input images');
      }

      const newImages = [];
      for (let file of files) {
        const base64 = await fileToBase64(file);
        newImages.push({
          id: Date.now() + Math.random(),
          file,
          base64
        });
      }

      setInputImages(prev => [...prev, ...newImages].slice(0, 5));
    } catch (err) {
      setError(err.message);
    }
  };

  const processSingleFile = async (files) => {
    try {
      const file = files[0];
      if (!file) return;
      
      validateImageFile(file);
      const base64 = await fileToBase64(file);
      setEditImage({ file, base64 });
    } catch (err) {
      setError(err.message);
    }
  };

  const handleMultiImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    await processMultiFiles(files);
    e.target.value = '';
  };

  const handleSingleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    await processSingleFile(files);
    e.target.value = '';
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length === 0) {
      setError('Chỉ chấp nhận file ảnh');
      return;
    }

    if (mode === 'in-context') {
      await processMultiFiles(imageFiles);
    } else {
      await processSingleFile(imageFiles);
    }
  };

  const removeInputImage = (id) => {
    setInputImages(prev => prev.filter(img => img.id !== id));
  };

  const handleGenerate = async () => {
    if (!formData.instruction.trim()) {
      setError('Vui lòng nhập instruction');
      return;
    }

    if (mode === 'in-context' && inputImages.length === 0) {
      setError('Vui lòng upload ít nhất 1 input image');
      return;
    }

    if (mode === 'edit' && !editImage) {
      setError('Vui lòng upload ảnh cần chỉnh sửa');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;
      
      if (mode === 'in-context') {
        const requestData = {
          instruction: formData.instruction,
          input_images: inputImages.map(img => img.base64),
          width: formData.width,
          height: formData.height,
          num_inference_steps: formData.num_inference_steps,
          text_guidance_scale: formData.text_guidance_scale,
          image_guidance_scale: formData.image_guidance_scale,
          negative_prompt: formData.negative_prompt || "",
          seed: formData.seed === 0 ? Math.floor(Math.random() * 2147483647) : formData.seed,
          scheduler: formData.scheduler
        };
        response = await omnigen2InContextGeneration(requestData);
      } else {
        const requestData = {
          instruction: formData.instruction,
          input_image: editImage.base64,
          width: formData.width,
          height: formData.height,
          num_inference_steps: formData.num_inference_steps,
          text_guidance_scale: formData.text_guidance_scale,
          image_guidance_scale: formData.image_guidance_scale,
          negative_prompt: formData.negative_prompt || "",
          seed: formData.seed === 0 ? Math.floor(Math.random() * 2147483647) : formData.seed,
          scheduler: formData.scheduler
        };
        response = await omnigen2EditImage(requestData);
      }

      setResult(response);
      
      // Call callback to save image to gallery
      if (onImageGenerated && response.image) {
        onImageGenerated(response.image);
      }
    } catch (err) {
      setError(err.message || 'Có lỗi xảy ra khi xử lý ảnh');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="section">
      <div className="section-header">
        <h2>🤖 OmniGen2 - Advanced AI Image Generation & Editing</h2>
        <button 
          className="config-button"
          onClick={onConfigOpen}
          title="Cấu hình"
        >
          ☰
        </button>
      </div>
      
      {error && <div className="error">{error}</div>}
      
      {/* Mode Selection */}
      <div className="nav" style={{ marginBottom: '20px' }}>
        <button
          className={`nav-button ${mode === 'in-context' ? 'active' : ''}`}
          onClick={() => handleModeChange('in-context')}
        >
          In-Context Generation
        </button>
        <button
          className={`nav-button ${mode === 'edit' ? 'active' : ''}`}
          onClick={() => handleModeChange('edit')}
        >
          Image Editing
        </button>
      </div>

      <div className="two-column">
        {/* Input Form - Chỉ giữ lại instruction */}
        <div>
          <div className="form-group">
            <label>Instruction *</label>
            <textarea
              name="instruction"
              value={formData.instruction}
              onChange={handleInputChange}
              placeholder={
                mode === 'in-context' 
                  ? "VD: Let the person in image 2 hold the toy from image 1 in a parking lot"
                  : "VD: Change the background to classroom, Add a hat, Remove the cat"
              }
              rows={3}
            />
          </div>

          {/* Image Upload based on mode */}
          {mode === 'in-context' ? (
            <div className="form-group">
              <label>Input Images (Tối đa 5)</label>
              <div 
                className={`file-input-wrapper ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <label className="file-input">
                  <input
                    ref={multiFileInputRef}
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleMultiImageUpload}
                  />
                  <span>
                    {isDragOver 
                      ? 'Thả ảnh vào đây' 
                      : 'Chọn nhiều ảnh hoặc kéo thả để tạo composition'
                    }
                  </span>
                </label>
              </div>
            </div>
          ) : (
            <div className="form-group">
              <label>Image để chỉnh sửa</label>
              <div 
                className={`file-input-wrapper ${isDragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <label className="file-input">
                  <input
                    ref={singleFileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleSingleImageUpload}
                  />
                  <span>
                    {isDragOver 
                      ? 'Thả ảnh vào đây' 
                      : 'Chọn ảnh hoặc kéo thả để chỉnh sửa'
                    }
                  </span>
                </label>
              </div>
            </div>
          )}

          <button 
            className="button" 
            onClick={handleGenerate}
            disabled={loading || (mode === 'in-context' ? inputImages.length === 0 : !editImage)}
          >
            {loading ? 'Đang xử lý...' : (mode === 'in-context' ? 'Tạo Composition' : 'Chỉnh sửa ảnh')}
          </button>
        </div>

        {/* Image Preview */}
        <div>
          {mode === 'in-context' ? (
            <>
              <h3>Input Images ({inputImages.length}/5)</h3>
              {inputImages.length === 0 ? (
                <div className="info">Chưa có input images</div>
              ) : (
                <div className="grid">
                  {inputImages.map((img, index) => (
                    <div key={img.id} className="card">
                      <img 
                        src={img.base64} 
                        alt={`Input ${index + 1}`}
                        style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '8px' }}
                      />
                      <div style={{ marginTop: '10px', textAlign: 'center' }}>
                        <p>Image {index + 1}</p>
                        <button 
                          className="button danger"
                          onClick={() => removeInputImage(img.id)}
                          style={{ width: '100%', padding: '8px' }}
                        >
                          Xóa
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          ) : (
            <>
              <h3>Image để chỉnh sửa</h3>
              {!editImage ? (
                <div className="info">Chưa chọn ảnh</div>
              ) : (
                <div className="card">
                  <img 
                    src={editImage.base64} 
                    alt="Edit target"
                    style={{ width: '100%', borderRadius: '8px' }}
                  />
                  <button 
                    className="button danger"
                    onClick={() => setEditImage(null)}
                    style={{ marginTop: '10px', width: '100%' }}
                  >
                    Xóa ảnh
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Đang xử lý, vui lòng đợi...</p>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="result-container">
          <h3>Kết quả</h3>
          <div className="two-column">
            <div>
              <img 
                src={result.image} 
                alt="Generated"
                className="result-image"
              />
            </div>
            <div>
              <p><strong>Instruction:</strong> {result.instruction}</p>
              <p><strong>Seed:</strong> {result.seed}</p>
              <p><strong>Input Images:</strong> {result.num_input_images}</p>
              <p><strong>Generation Time:</strong> {result.generation_time?.toFixed(2)}s</p>
              
              {result.individual_images && result.individual_images.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4>Individual Results</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '10px' }}>
                    {result.individual_images.map((img, index) => (
                      <img 
                        key={index}
                        src={img} 
                        alt={`Result ${index}`}
                        style={{ width: '100%', borderRadius: '4px' }}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default OmniGen2Component; 