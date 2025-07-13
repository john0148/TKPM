import React from 'react';

const DreamOConfigPopup = ({ isOpen, onClose, formData, onInputChange }) => {
  if (!isOpen) return null;

  return (
    <div className="config-overlay" onClick={onClose}>
      <div className="config-popup" onClick={(e) => e.stopPropagation()}>
        <div className="config-header">
          <h3>⚙️ Cấu hình DreamO</h3>
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
              name="neg_prompt"
              value={formData.neg_prompt}
              onChange={onInputChange}
              placeholder="Những gì bạn không muốn trong ảnh..."
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
              <label>Steps</label>
              <input
                type="number"
                name="num_steps"
                value={formData.num_steps}
                onChange={onInputChange}
                min="1"
                max="50"
              />
            </div>
            <div className="form-group">
              <label>Guidance</label>
              <input
                type="number"
                name="guidance"
                value={formData.guidance}
                onChange={onInputChange}
                min="1"
                max="20"
                step="0.1"
              />
            </div>
          </div>

          <div className="grid">
            <div className="form-group">
              <label>Reference Resolution</label>
              <input
                type="number"
                name="ref_res"
                value={formData.ref_res}
                onChange={onInputChange}
                min="256"
                max="1024"
                step="64"
              />
            </div>
            <div className="form-group">
              <label>Seed (-1 = random)</label>
              <input
                type="number"
                name="seed"
                value={formData.seed}
                onChange={onInputChange}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DreamOConfigPopup; 