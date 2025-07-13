import React from 'react';

const OmniGen2ConfigPopup = ({ isOpen, onClose, formData, onInputChange }) => {
  if (!isOpen) return null;

  return (
    <div className="config-overlay" onClick={onClose}>
      <div className="config-popup" onClick={(e) => e.stopPropagation()}>
        <div className="config-header">
          <h3>⚙️ Cấu hình OmniGen2</h3>
          <button 
            className="close-button"
            onClick={onClose}
          >
            ✕
          </button>
        </div>
        
        <div className="config-content">
          <div className="form-group">
            <label>Negative Prompt</label>
            <textarea
              name="negative_prompt"
              value={formData.negative_prompt}
              onChange={onInputChange}
              placeholder="Những gì bạn không muốn..."
              rows={2}
            />
          </div>

          <div className="grid">
            <div className="form-group">
              <label>Width</label>
              <input
                type="number"
                name="width"
                value={formData.width}
                onChange={onInputChange}
                min="512"
                max="2048"
                step="64"
              />
            </div>
            <div className="form-group">
              <label>Height</label>
              <input
                type="number"
                name="height"
                value={formData.height}
                onChange={onInputChange}
                min="512"
                max="2048"
                step="64"
              />
            </div>
          </div>

          <div className="grid">
            <div className="form-group">
              <label>Inference Steps</label>
              <input
                type="number"
                name="num_inference_steps"
                value={formData.num_inference_steps}
                onChange={onInputChange}
                min="1"
                max="100"
              />
            </div>
            <div className="form-group">
              <label>Scheduler</label>
              <select
                name="scheduler"
                value={formData.scheduler}
                onChange={onInputChange}
              >
                <option value="euler">Euler</option>
                <option value="dpmsolver">DPM Solver</option>
              </select>
            </div>
          </div>

          <div className="grid">
            <div className="form-group">
              <label>Text Guidance</label>
              <input
                type="number"
                name="text_guidance_scale"
                value={formData.text_guidance_scale}
                onChange={onInputChange}
                min="1"
                max="20"
                step="0.1"
              />
            </div>
            <div className="form-group">
              <label>Image Guidance</label>
              <input
                type="number"
                name="image_guidance_scale"
                value={formData.image_guidance_scale}
                onChange={onInputChange}
                min="1"
                max="10"
                step="0.1"
              />
            </div>
          </div>

          <div className="form-group">
            <label>Seed (0 = random)</label>
            <input
              type="number"
              name="seed"
              value={formData.seed}
              onChange={onInputChange}
              min="0"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default OmniGen2ConfigPopup; 