import React, { useState, useEffect } from 'react';
import { 
  startTraining, 
  getTrainingStatus, 
  listTrainingSessions, 
  deleteTrainingSession,
  inferenceWithTrainedModel,
  validateImageFile 
} from '../utils/api';

const TrainingComponent = () => {
  const [mode, setMode] = useState('new'); // 'new', 'manage', 'inference'
  const [formData, setFormData] = useState({
    nameObject: '',
    description: '',
    subjectType: 'object' // mặc định là object
  });
  const [referenceImages, setReferenceImages] = useState([]);
  const [trainingSessions, setTrainingSessions] = useState([]);
  const [currentTraining, setCurrentTraining] = useState(null);
  const [inferenceData, setInferenceData] = useState({
    trainingId: '',
    prompt: '',
    negative_prompt: '',
    num_inference_steps: 20,
    guidance_scale: 4.0
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [result, setResult] = useState(null);

  // Polling interval for training status
  useEffect(() => {
    let interval;
    if (currentTraining && ['generating_variations', 'preparing_dataset', 'training', 'converting', 'generating'].includes(currentTraining.status)) {
      interval = setInterval(() => {
        checkTrainingStatus(currentTraining.training_id);
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [currentTraining]);

  // Load training sessions on component mount and mode change
  useEffect(() => {
    if (mode === 'manage' || mode === 'inference') {
      loadTrainingSessions();
    }
  }, [mode]);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleInferenceInputChange = (e) => {
    const { name, value, type } = e.target;
    setInferenceData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleImageUpload = async (e) => {
    const files = Array.from(e.target.files);
    
    try {
      files.forEach(file => validateImageFile(file));
      
      if (files.length > 5) {
        throw new Error('Tối đa 5 reference images');
      }

      const newImages = files.map(file => ({
        id: Date.now() + Math.random(),
        file,
        preview: URL.createObjectURL(file)
      }));

      setReferenceImages(prev => [...prev, ...newImages].slice(0, 5));
      e.target.value = '';
    } catch (err) {
      setError(err.message);
    }
  };

  const removeImage = (id) => {
    setReferenceImages(prev => {
      const toRemove = prev.find(img => img.id === id);
      if (toRemove?.preview) {
        URL.revokeObjectURL(toRemove.preview);
      }
      return prev.filter(img => img.id !== id);
    });
  };

  const handleStartTraining = async () => {
    if (!formData.nameObject.trim()) {
      setError('Vui lòng nhập tên object');
      return;
    }

    if (referenceImages.length === 0) {
      setError('Vui lòng upload ít nhất 1 reference image');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const files = referenceImages.map(img => img.file);
      const response = await startTraining(formData.nameObject, formData.description, files, formData.subjectType);
      
      setSuccess('Training pipeline đã bắt đầu!');
      setCurrentTraining({
        training_id: response.training_id,
        status: 'initialized',
        progress: 0,
        message: response.message
      });
      
      // Start checking status
      setTimeout(() => checkTrainingStatus(response.training_id), 2000);
      
    } catch (err) {
      setError(err.message || 'Có lỗi xảy ra khi bắt đầu training');
    } finally {
      setLoading(false);
    }
  };

  const checkTrainingStatus = async (trainingId) => {
    try {
      const status = await getTrainingStatus(trainingId);
      setCurrentTraining(status);
      
      if (status.status === 'completed') {
        setSuccess('Training hoàn thành thành công!');
      } else if (status.status === 'failed') {
        setError(`Training thất bại: ${status.message}`);
      }
    } catch (err) {
      console.error('Error checking training status:', err);
    }
  };

  const loadTrainingSessions = async () => {
    try {
      const response = await listTrainingSessions();
      setTrainingSessions(response.sessions || []);
    } catch (err) {
      setError('Không thể tải danh sách training sessions');
    }
  };

  const handleDeleteSession = async (trainingId) => {
    if (!confirm('Bạn có chắc muốn xóa training session này?')) return;

    try {
      await deleteTrainingSession(trainingId);
      setSuccess('Đã xóa training session');
      loadTrainingSessions();
    } catch (err) {
      setError('Không thể xóa training session');
    }
  };

  const handleInference = async () => {
    if (!inferenceData.trainingId) {
      setError('Vui lòng chọn trained model');
      return;
    }

    if (!inferenceData.prompt.trim()) {
      setError('Vui lòng nhập prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await inferenceWithTrainedModel(inferenceData.trainingId, {
        prompt: inferenceData.prompt,
        negative_prompt: inferenceData.negative_prompt || "",
        num_inference_steps: inferenceData.num_inference_steps,
        guidance_scale: inferenceData.guidance_scale
      });

      setResult(response);
      setSuccess('Tạo ảnh thành công!');
    } catch (err) {
      setError(err.message || 'Có lỗi xảy ra khi tạo ảnh');
    } finally {
      setLoading(false);
    }
  };

  const getStatusBadgeClass = (status) => {
    const statusMap = {
      'initialized': 'pending',
      'generating_variations': 'generating',
      'preparing_dataset': 'generating',
      'training': 'training',
      'converting': 'training',
      'generating': 'generating',
      'completed': 'completed',
      'failed': 'failed'
    };
    return `status-badge ${statusMap[status] || 'pending'}`;
  };

  return (
    <div className="section">
      <h2>🎓 Training - Huấn luyện Model tùy chỉnh</h2>
      
      {error && <div className="error">{error}</div>}
      {success && <div className="success">{success}</div>}
      
      {/* Mode Selection */}
      <div className="nav" style={{ marginBottom: '20px' }}>
        <button
          className={`nav-button ${mode === 'new' ? 'active' : ''}`}
          onClick={() => setMode('new')}
        >
          New Training
        </button>
        <button
          className={`nav-button ${mode === 'manage' ? 'active' : ''}`}
          onClick={() => setMode('manage')}
        >
          Manage Sessions
        </button>
        <button
          className={`nav-button ${mode === 'inference' ? 'active' : ''}`}
          onClick={() => setMode('inference')}
        >
          Inference
        </button>
      </div>

      {/* New Training */}
      {mode === 'new' && (
        <div className="two-column">
          <div>
            <div className="form-group">
              <label>Tên Object *</label>
              <input
                type="text"
                name="nameObject"
                value={formData.nameObject}
                onChange={handleInputChange}
                placeholder="VD: my_cat, john_person, special_toy..."
              />
            </div>

            <div className="form-group subject-type-group">
              <label className="subject-type-label">Loại chủ thể *</label>
              <div className="subject-type-btn-group">
                <label className={`subject-type-btn${formData.subjectType === 'object' ? ' active' : ''}`}> 
                  <input
                    type="radio"
                    name="subjectType"
                    value="object"
                    checked={formData.subjectType === 'object'}
                    onChange={() => setFormData(prev => ({ ...prev, subjectType: 'object' }))}
                  />
                  Object
                </label>
                <label className={`subject-type-btn${formData.subjectType === 'background' ? ' active' : ''}`}> 
                  <input
                    type="radio"
                    name="subjectType"
                    value="background"
                    checked={formData.subjectType === 'background'}
                    onChange={() => setFormData(prev => ({ ...prev, subjectType: 'background' }))}
                  />
                  Background
                </label>
              </div>
            </div>

            <div className="form-group">
              <label>Mô tả (tùy chọn)</label>
              <textarea
                name="description"
                value={formData.description}
                onChange={handleInputChange}
                placeholder="Mô tả chi tiết về object..."
                rows={3}
              />
            </div>

            <div className="form-group">
              <label>Reference Images (1-5 ảnh) *</label>
              <div className="file-input-wrapper">
                <label className="file-input">
                  <input
                    type="file"
                    multiple
                    accept="image/*"
                    onChange={handleImageUpload}
                  />
                  <span>Chọn 1-5 ảnh reference của object</span>
                </label>
              </div>
            </div>

            <button 
              className="button" 
              onClick={handleStartTraining}
              disabled={loading || referenceImages.length === 0}
            >
              {loading ? 'Đang bắt đầu...' : 'Bắt đầu Training'}
            </button>

            <div className="info" style={{ marginTop: '20px' }}>
              <h4>Training Pipeline:</h4>
              <p>1. Generate 15 variations từ reference images (DreamO)</p>
              <p>2. Chuẩn bị dataset cho training</p>
              <p>3. Train LoRA model với OmniGen2</p>
              <p>4. Generate test image</p>
              <p><strong>Thời gian ước tính:</strong> 15-30 phút</p>
            </div>
          </div>

          <div>
            <h3>Reference Images ({referenceImages.length}/5)</h3>
            {referenceImages.length === 0 ? (
              <div className="info">Chưa có reference images</div>
            ) : (
              <div className="grid">
                {referenceImages.map(img => (
                  <div key={img.id} className="card">
                    <img 
                      src={img.preview} 
                      alt="Reference"
                      style={{ width: '100%', height: '150px', objectFit: 'cover', borderRadius: '8px' }}
                    />
                    <button 
                      className="button danger"
                      onClick={() => removeImage(img.id)}
                      style={{ marginTop: '10px', width: '100%', padding: '8px' }}
                    >
                      Xóa
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Current Training Status */}
            {currentTraining && (
              <div className="card" style={{ marginTop: '20px' }}>
                <h4>Training Status</h4>
                <p><strong>ID:</strong> {currentTraining.training_id}</p>
                <p>
                  <strong>Status:</strong> 
                  <span className={getStatusBadgeClass(currentTraining.status)}>
                    {currentTraining.status}
                  </span>
                </p>
                <p><strong>Message:</strong> {currentTraining.message}</p>
                
                {currentTraining.progress !== undefined && (
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${currentTraining.progress * 100}%` }}
                    ></div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Manage Sessions */}
      {mode === 'manage' && (
        <div>
          <h3>Training Sessions</h3>
          {trainingSessions.length === 0 ? (
            <div className="info">Không có training sessions nào</div>
          ) : (
            <div className="grid">
              {trainingSessions.map(session => (
                <div key={session.training_id} className="card">
                  <h4>{session.name_object}</h4>
                  <p><strong>ID:</strong> {session.training_id.slice(0, 8)}...</p>
                  <p><strong>Created:</strong> {new Date(session.created_at).toLocaleString()}</p>
                  <p>
                    <strong>Status:</strong>
                    <span className={getStatusBadgeClass(session.status)}>
                      {session.status}
                    </span>
                  </p>
                  {session.description && <p><strong>Description:</strong> {session.description}</p>}
                  
                  <div style={{ marginTop: '10px' }}>
                    <button 
                      className="button danger"
                      onClick={() => handleDeleteSession(session.training_id)}
                      style={{ width: '100%' }}
                    >
                      Xóa
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Inference */}
      {mode === 'inference' && (
        <div className="two-column">
          <div>
            <div className="form-group">
              <label>Chọn Trained Model *</label>
              <select
                name="trainingId"
                value={inferenceData.trainingId}
                onChange={handleInferenceInputChange}
              >
                <option value="">-- Chọn model --</option>
                {trainingSessions
                  .filter(session => session.status === 'completed')
                  .map(session => (
                    <option key={session.training_id} value={session.training_id}>
                      {session.name_object} ({session.training_id.slice(0, 8)}...)
                    </option>
                  ))}
              </select>
            </div>

            <div className="form-group">
              <label>Prompt *</label>
              <textarea
                name="prompt"
                value={inferenceData.prompt}
                onChange={handleInferenceInputChange}
                placeholder="VD: A professional photo of [tên_object], high quality, detailed"
                rows={3}
              />
            </div>

            <div className="form-group">
              <label>Negative Prompt</label>
              <textarea
                name="negative_prompt"
                value={inferenceData.negative_prompt}
                onChange={handleInferenceInputChange}
                placeholder="Những gì bạn không muốn..."
                rows={2}
              />
            </div>

            <div className="grid">
              <div className="form-group">
                <label>Inference Steps</label>
                <input
                  type="number"
                  name="num_inference_steps"
                  value={inferenceData.num_inference_steps}
                  onChange={handleInferenceInputChange}
                  min="1"
                  max="100"
                />
              </div>
              <div className="form-group">
                <label>Guidance Scale</label>
                <input
                  type="number"
                  name="guidance_scale"
                  value={inferenceData.guidance_scale}
                  onChange={handleInferenceInputChange}
                  min="1"
                  max="20"
                  step="0.1"
                />
              </div>
            </div>

            <button 
              className="button" 
              onClick={handleInference}
              disabled={loading || !inferenceData.trainingId || !inferenceData.prompt.trim()}
            >
              {loading ? 'Đang tạo...' : 'Tạo ảnh với Trained Model'}
            </button>
          </div>

          <div>
            <h3>Available Trained Models</h3>
            {trainingSessions.filter(s => s.status === 'completed').length === 0 ? (
              <div className="info">Chưa có trained models nào</div>
            ) : (
              <div>
                {trainingSessions
                  .filter(session => session.status === 'completed')
                  .map(session => (
                    <div key={session.training_id} className="card">
                      <h4>{session.name_object}</h4>
                      <p><strong>ID:</strong> {session.training_id.slice(0, 8)}...</p>
                      <p><strong>Completed:</strong> {new Date(session.created_at).toLocaleString()}</p>
                      {session.description && <p>{session.description}</p>}
                    </div>
                  ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Đang xử lý, vui lòng đợi...</p>
        </div>
      )}

      {/* Inference Result */}
      {result && (
        <div className="result-container">
          <h3>Kết quả Inference</h3>
          <div className="two-column">
            <div>
              <img 
                src={`data:image/png;base64,${result.image_path.split(',')[1] || result.image_path}`} 
                alt="Generated with trained model"
                className="result-image"
              />
            </div>
            <div>
              <p><strong>Prompt:</strong> {inferenceData.prompt}</p>
              <p><strong>Training ID:</strong> {result.training_id}</p>
              <p><strong>Image Path:</strong> {result.image_path}</p>
              <p><strong>Success:</strong> {result.success ? 'Yes' : 'No'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingComponent; 