import React, { useState, useEffect } from 'react';
import { 
  healthCheck, 
  dreamoHealthCheck, 
  omnigen2HealthCheck,
  getDreamoModelInfo,
  getOmnigen2ModelInfo 
} from '../utils/api';

const HealthComponent = ({ healthData, setHealthData }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [modelInfo, setModelInfo] = useState({
    dreamo: null,
    omnigen2: null
  });

  useEffect(() => {
    checkAllHealth();
    loadModelInfo();
  }, []);

  const checkAllHealth = async () => {
    setLoading(true);
    setError(null);

    try {
      const [general, dreamoHealth, omnigen2Health] = await Promise.allSettled([
        healthCheck(),
        dreamoHealthCheck(),
        omnigen2HealthCheck()
      ]);

      const healthResults = {
        general: general.status === 'fulfilled' ? general.value : { healthy: false, error: general.reason?.message },
        dreamo: dreamoHealth.status === 'fulfilled' ? dreamoHealth.value : { healthy: false, error: dreamoHealth.reason?.message },
        omnigen2: omnigen2Health.status === 'fulfilled' ? omnigen2Health.value : { healthy: false, error: omnigen2Health.reason?.message },
        lastChecked: new Date().toISOString()
      };

      setHealthData(healthResults);
    } catch (err) {
      setError('Không thể kiểm tra trạng thái hệ thống');
    } finally {
      setLoading(false);
    }
  };

  const loadModelInfo = async () => {
    try {
      const [dreamoInfo, omnigen2Info] = await Promise.allSettled([
        getDreamoModelInfo(),
        getOmnigen2ModelInfo()
      ]);

      setModelInfo({
        dreamo: dreamoInfo.status === 'fulfilled' ? dreamoInfo.value : null,
        omnigen2: omnigen2Info.status === 'fulfilled' ? omnigen2Info.value : null
      });
    } catch (err) {
      console.error('Error loading model info:', err);
    }
  };

  const formatMemoryUsage = (memoryStr) => {
    if (!memoryStr) return 'N/A';
    return memoryStr;
  };

  const getHealthIndicator = (healthy) => {
    return (
      <span className={`health-indicator ${healthy ? 'healthy' : 'unhealthy'}`}></span>
    );
  };

  return (
    <div className="section">
      <h2>💊 Health - Trạng thái hệ thống</h2>
      
      {error && <div className="error">{error}</div>}
      
      <div style={{ marginBottom: '20px' }}>
        <button 
          className="button" 
          onClick={checkAllHealth}
          disabled={loading}
        >
          {loading ? 'Đang kiểm tra...' : 'Refresh Health Status'}
        </button>
        
        {healthData?.lastChecked && (
          <p style={{ marginTop: '10px', color: '#666' }}>
            Cập nhật lần cuối: {new Date(healthData.lastChecked).toLocaleString()}
          </p>
        )}
      </div>

      {loading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Đang kiểm tra trạng thái hệ thống...</p>
        </div>
      )}

      {healthData && (
        <div className="grid">
          {/* General Health */}
          <div className="card">
            <h3>🌐 General System</h3>
            <p>
              {getHealthIndicator(healthData.general?.healthy)}
              <strong>Status:</strong> {healthData.general?.healthy ? 'Healthy' : 'Unhealthy'}
            </p>
            {healthData.general?.models && (
              <div>
                <p><strong>Models Loaded:</strong></p>
                <ul>
                  <li>DreamO: {healthData.general.models.dreamo ? '✅' : '❌'}</li>
                  <li>OmniGen2: {healthData.general.models.omnigen2 ? '✅' : '❌'}</li>
                </ul>
              </div>
            )}
            {healthData.general?.error && (
              <p style={{ color: 'red' }}>Error: {healthData.general.error}</p>
            )}
          </div>

          {/* DreamO Health */}
          <div className="card">
            <h3>🎨 DreamO Model</h3>
            <p>
              {getHealthIndicator(healthData.dreamo?.healthy)}
              <strong>Status:</strong> {healthData.dreamo?.healthy ? 'Healthy' : 'Unhealthy'}
            </p>
            {healthData.dreamo?.healthy && (
              <div>
                <p><strong>Version:</strong> {healthData.dreamo.version}</p>
                <p><strong>Device:</strong> {healthData.dreamo.device}</p>
                <p><strong>Model Loaded:</strong> {healthData.dreamo.model_loaded ? 'Yes' : 'No'}</p>
              </div>
            )}
            {healthData.dreamo?.error && (
              <p style={{ color: 'red' }}>Error: {healthData.dreamo.error}</p>
            )}

            {/* DreamO Model Info */}
            {modelInfo.dreamo && (
              <div style={{ marginTop: '15px', fontSize: '0.9rem' }}>
                <h4>Model Information</h4>
                <p><strong>Description:</strong> {modelInfo.dreamo.description}</p>
                <p><strong>Optimization:</strong> {modelInfo.dreamo.optimization}</p>
                <p><strong>Typical Inference Time:</strong> {modelInfo.dreamo.typical_inference_time}</p>
                
                {modelInfo.dreamo.memory_requirements && (
                  <div>
                    <p><strong>Memory Requirements:</strong></p>
                    <ul style={{ fontSize: '0.8rem' }}>
                      <li>Nunchaku: {modelInfo.dreamo.memory_requirements.nunchaku_mode}</li>
                      <li>INT8: {modelInfo.dreamo.memory_requirements.int8_mode}</li>
                      <li>Full: {modelInfo.dreamo.memory_requirements.full_precision}</li>
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* OmniGen2 Health */}
          <div className="card">
            <h3>🤖 OmniGen2 Model</h3>
            <p>
              {getHealthIndicator(healthData.omnigen2?.healthy)}
              <strong>Status:</strong> {healthData.omnigen2?.healthy ? 'Healthy' : 'Unhealthy'}
            </p>
            {healthData.omnigen2?.healthy && (
              <div>
                <p><strong>Device:</strong> {healthData.omnigen2.device}</p>
                <p><strong>Model Loaded:</strong> {healthData.omnigen2.model_loaded ? 'Yes' : 'No'}</p>
                {healthData.omnigen2.memory_usage && (
                  <p><strong>Memory Usage:</strong> {formatMemoryUsage(healthData.omnigen2.memory_usage)}</p>
                )}
              </div>
            )}
            {healthData.omnigen2?.error && (
              <p style={{ color: 'red' }}>Error: {healthData.omnigen2.error}</p>
            )}

            {/* OmniGen2 Model Info */}
            {modelInfo.omnigen2 && (
              <div style={{ marginTop: '15px', fontSize: '0.9rem' }}>
                <h4>Model Information</h4>
                <p><strong>Description:</strong> {modelInfo.omnigen2.description}</p>
                <p><strong>Optimization:</strong> {modelInfo.omnigen2.optimization}</p>
                <p><strong>Typical Inference Time:</strong> {modelInfo.omnigen2.typical_inference_time}</p>
                
                {modelInfo.omnigen2.memory_requirements && (
                  <div>
                    <p><strong>Memory Requirements:</strong></p>
                    <ul style={{ fontSize: '0.8rem' }}>
                      <li>DFloat11: {modelInfo.omnigen2.memory_requirements.dfloat11_compressed}</li>
                      <li>Original: {modelInfo.omnigen2.memory_requirements.original_bfloat16}</li>
                      <li>Compression: {modelInfo.omnigen2.memory_requirements.compression_ratio}</li>
                    </ul>
                  </div>
                )}

                {modelInfo.omnigen2.architecture && (
                  <div>
                    <p><strong>Architecture:</strong></p>
                    <ul style={{ fontSize: '0.8rem' }}>
                      <li>Text Encoder: {modelInfo.omnigen2.architecture.text_encoder}</li>
                      <li>Transformer: {modelInfo.omnigen2.architecture.transformer}</li>
                      <li>VAE: {modelInfo.omnigen2.architecture.vae}</li>
                      <li>Scheduler: {modelInfo.omnigen2.architecture.scheduler}</li>
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Capabilities Overview */}
      {(modelInfo.dreamo || modelInfo.omnigen2) && (
        <div style={{ marginTop: '30px' }}>
          <h3>🚀 Capabilities Overview</h3>
          <div className="two-column">
            {modelInfo.dreamo && (
              <div className="card">
                <h4>DreamO Capabilities</h4>
                <ul>
                  {modelInfo.dreamo.capabilities?.map((capability, index) => (
                    <li key={index}>{capability}</li>
                  ))}
                </ul>
                <p><strong>Supported Tasks:</strong> IP, ID, Style</p>
                <p><strong>Max Reference Images:</strong> 10</p>
                <p><strong>Max Resolution:</strong> 2048x2048</p>
              </div>
            )}

            {modelInfo.omnigen2 && (
              <div className="card">
                <h4>OmniGen2 Capabilities</h4>
                <ul>
                  {modelInfo.omnigen2.capabilities?.map((capability, index) => (
                    <li key={index}>{capability}</li>
                  ))}
                </ul>
                <p><strong>Supported Tasks:</strong></p>
                <ul>
                  {modelInfo.omnigen2.supported_tasks?.map((task, index) => (
                    <li key={index}>{task}</li>
                  ))}
                </ul>
                <p><strong>Max Resolution:</strong> 2048x2048</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* System Recommendations */}
      <div style={{ marginTop: '30px' }}>
        <div className="info">
          <h4>💡 System Recommendations</h4>
          <ul>
            <li><strong>GPU Memory:</strong> Khuyến nghị tối thiểu 16GB VRAM cho cả 2 models</li>
            <li><strong>System RAM:</strong> Tối thiểu 32GB RAM</li>
            <li><strong>Storage:</strong> Models chiếm khoảng 30GB dung lượng</li>
            <li><strong>Performance:</strong> RTX 3080+ hoặc A100 để có hiệu năng tốt nhất</li>
            <li><strong>Monitoring:</strong> Kiểm tra health status thường xuyên để đảm bảo ổn định</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default HealthComponent; 