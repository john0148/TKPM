import React, { useState } from 'react';
import { dreamoGenerate, fileToBase64, validateImageFile } from '../utils/api';

const DreamOComponent = ({ formData, onFormDataChange, onConfigOpen }) => {
  const [refImages, setRefImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    onFormDataChange(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    
    try {
      // Validate files
      files.forEach(file => validateImageFile(file));
      
      if (files.length > 10) {
        throw new Error('Tối đa 10 reference images');
      }

      const newImages = [];
      for (let file of files) {
        const base64 = await fileToBase64(file);
        newImages.push({
          id: Date.now() + Math.random(),
          file,
          base64,
          task: 'ip' // default task
        });
      }

      setRefImages(prev => [...prev, ...newImages].slice(0, 10));
      e.target.value = ''; // Reset input
    } catch (err) {
      setError(err.message);
    }
  };

  const removeImage = (id) => {
    setRefImages(prev => prev.filter(img => img.id !== id));
  };

  const updateImageTask = (id, task) => {
    setRefImages(prev => prev.map(img => 
      img.id === id ? { ...img, task } : img
    ));
  };

  const handleGenerate = async () => {
    if (!formData.prompt.trim()) {
      setError('Vui lòng nhập prompt');
      return;
    }

    if (refImages.length === 0) {
      setError('Vui lòng upload ít nhất 1 reference image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const requestData = {
        prompt: formData.prompt,
        ref_images: refImages.map(img => ({
          image_data: img.base64,
          task: img.task
        })),
        width: formData.width,
        height: formData.height,
        num_steps: formData.num_steps,
        guidance: formData.guidance,
        seed: formData.seed === -1 ? -1 : formData.seed,
        neg_prompt: formData.neg_prompt || "",
        ref_res: formData.ref_res
      };

      console.log('Sending DreamO request:', requestData);
      const response = await dreamoGenerate(requestData);
      setResult(response);
    } catch (err) {
      setError(err.message || 'Có lỗi xảy ra khi tạo ảnh');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="section">
      <div className="section-header">
        <h2>🎨 DreamO - Image Generation với Reference Images</h2>
        <button 
          className="config-button"
          onClick={onConfigOpen}
          title="Cấu hình"
        >
          ☰
        </button>
      </div>
      
      {error && <div className="error">{error}</div>}
      
      <div className="two-column">
        {/* Input Form - Chỉ giữ lại prompt */}
        <div>
          <div className="form-group">
            <label>Prompt *</label>
            <textarea
              name="prompt"
              value={formData.prompt}
              onChange={handleInputChange}
              placeholder="Mô tả hình ảnh bạn muốn tạo..."
              rows={3}
            />
          </div>

          <div className="form-group">
            <label>Reference Images (Tối đa 10)</label>
            <div className="file-input-wrapper">
              <label className="file-input">
                <input
                  type="file"
                  multiple
                  accept="image/*"
                  onChange={handleImageUpload}
                />
                <span>Nhấn để chọn hoặc kéo thả ảnh vào đây</span>
              </label>
            </div>
          </div>

          <button 
            className="button" 
            onClick={handleGenerate}
            disabled={loading || refImages.length === 0}
          >
            {loading ? 'Đang tạo...' : 'Tạo ảnh'}
          </button>
        </div>

        {/* Reference Images Preview */}
        <div>
          <h3>Reference Images ({refImages.length}/10)</h3>
          {refImages.length === 0 ? (
            <div className="info">Chưa có reference images</div>
          ) : (
            <div className="grid">
              {refImages.map(img => (
                <div key={img.id} className="card">
                  <img 
                    src={img.base64} 
                    alt="Reference"
                    style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '8px' }}
                  />
                  <div style={{ marginTop: '10px' }}>
                    <select
                      value={img.task}
                      onChange={(e) => updateImageTask(img.id, e.target.value)}
                      style={{ width: '100%', marginBottom: '10px' }}
                    >
                      <option value="ip">IP - General</option>
                      <option value="id">ID - Face Identity</option>
                      <option value="style">Style - Style Transfer</option>
                    </select>
                    <button 
                      className="button danger"
                      onClick={() => removeImage(img.id)}
                      style={{ width: '100%', padding: '8px' }}
                    >
                      Xóa
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Đang tạo ảnh, vui lòng đợi...</p>
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
              <p><strong>Prompt:</strong> {result.prompt}</p>
              <p><strong>Seed:</strong> {result.seed}</p>
              <p><strong>Reference Count:</strong> {result.ref_count}</p>
              <p><strong>Generation Time:</strong> {result.generation_time?.toFixed(2)}s</p>
              
              {result.debug_images && result.debug_images.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <h4>Debug Images</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(100px, 1fr))', gap: '10px' }}>
                    {result.debug_images.map((img, index) => (
                      <img 
                        key={index}
                        src={img} 
                        alt={`Debug ${index}`}
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

export default DreamOComponent; 