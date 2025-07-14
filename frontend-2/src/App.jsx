import React, { useState, useEffect } from 'react';
import DreamOComponent from './components/DreamOComponent';
import OmniGen2Component from './components/OmniGen2Component';
import DreamOConfigPopup from './components/DreamOConfigPopup';
import OmniGen2ConfigPopup from './components/OmniGen2ConfigPopup';
import TrainingComponent from './components/TrainingComponent';
import HealthComponent from './components/HealthComponent';
import ImageGallery from './components/ImageGallery';

function App() {
  const [activeTab, setActiveTab] = useState('dreamo');
  const [healthData, setHealthData] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [configType, setConfigType] = useState(null);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [dreamoFormData, setDreamoFormData] = useState({
    prompt: '',
    width: 1024,
    height: 1024,
    num_steps: 12,
    guidance: 4.5,
    seed: -1,
    neg_prompt: '',
    ref_res: 768
  });
  const [omnigen2FormData, setOmnigen2FormData] = useState({
    instruction: '',
    width: 1024,
    height: 1024,
    num_inference_steps: 50,
    text_guidance_scale: 5.0,
    image_guidance_scale: 2.0,
    negative_prompt: '',
    seed: 0,
    scheduler: 'euler'
  });

  const tabs = [
    { id: 'dreamo', label: 'Stage 1', desc: 'Gen human with mull object' },
    { id: 'omnigen2', label: 'Stage 2', desc: 'Gen multihuman in the image' }
  ];

  const handleConfigOpen = (type) => {
    setConfigType(type);
    setShowConfig(true);
  };

  const handleConfigClose = () => {
    setShowConfig(false);
    setConfigType(null);
  };

  const handleDreamoInputChange = (e) => {
    const { name, value, type } = e.target;
    setDreamoFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleOmnigen2InputChange = (e) => {
    const { name, value, type } = e.target;
    setOmnigen2FormData(prev => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) : value
    }));
  };

  const handleImageGenerated = (imageData, model) => {
    const newImage = {
      id: Date.now() + Math.random(),
      url: imageData,
      prompt: model === 'dreamo' ? dreamoFormData.prompt : omnigen2FormData.instruction,
      model: model,
      timestamp: new Date().toISOString()
    };
    setGeneratedImages(prev => [newImage, ...prev]);
  };

  const handleDeleteImage = (imageId) => {
    setGeneratedImages(prev => prev.filter(img => img.id !== imageId));
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dreamo':
        return <DreamOComponent 
          formData={dreamoFormData}
          onFormDataChange={setDreamoFormData}
          onConfigOpen={() => handleConfigOpen('dreamo')}
          onImageGenerated={(imageData) => handleImageGenerated(imageData, 'dreamo')}
        />;
      case 'omnigen2':
        return <OmniGen2Component 
          formData={omnigen2FormData}
          onFormDataChange={setOmnigen2FormData}
          onConfigOpen={() => handleConfigOpen('omnigen2')}
          onImageGenerated={(imageData) => handleImageGenerated(imageData, 'omnigen2')}
        />;
      default:
        return <DreamOComponent 
          formData={dreamoFormData}
          onFormDataChange={setDreamoFormData}
          onConfigOpen={() => handleConfigOpen('dreamo')}
          onImageGenerated={(imageData) => handleImageGenerated(imageData, 'dreamo')}
        />;
    }
  };

  return (
    <div className={`app ${showConfig ? 'config-active' : ''}`}>
      <header className="header">
        <h1>🎨 AI Image Generation</h1>
        <p>Giao diện thống nhất cho DreamO và OmniGen2</p>
      </header>

      <nav className="nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`nav-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            title={tab.desc}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="main-layout">
        <aside className="gallery-sidebar">
          <ImageGallery 
            images={generatedImages}
            onDeleteImage={handleDeleteImage}
          />
        </aside>
        
        <main className="content">
          {renderTabContent()}
        </main>
      </div>

      {/* Config Popups */}
      {configType === 'dreamo' && (
        <DreamOConfigPopup
          isOpen={showConfig}
          onClose={handleConfigClose}
          formData={dreamoFormData}
          onInputChange={handleDreamoInputChange}
        />
      )}
      
      {configType === 'omnigen2' && (
        <OmniGen2ConfigPopup
          isOpen={showConfig}
          onClose={handleConfigClose}
          formData={omnigen2FormData}
          onInputChange={handleOmnigen2InputChange}
        />
      )}
    </div>
  );
}

export default App; 