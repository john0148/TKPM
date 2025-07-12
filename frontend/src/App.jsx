import React, { useState, useEffect } from 'react';
import DreamOComponent from './components/DreamOComponent';
import OmniGen2Component from './components/OmniGen2Component';
import TrainingComponent from './components/TrainingComponent';
import HealthComponent from './components/HealthComponent';

function App() {
  const [activeTab, setActiveTab] = useState('dreamo');
  const [healthData, setHealthData] = useState(null);

  const tabs = [
    { id: 'dreamo', label: 'DreamO', desc: 'Tạo ảnh với Reference Images' },
    { id: 'omnigen2', label: 'OmniGen2', desc: 'Tạo & Chỉnh sửa ảnh với AI' },
    { id: 'training', label: 'Training', desc: 'Huấn luyện Model tùy chỉnh' },
    { id: 'health', label: 'Health', desc: 'Trạng thái hệ thống' }
  ];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dreamo':
        return <DreamOComponent />;
      case 'omnigen2':
        return <OmniGen2Component />;
      case 'training':
        return <TrainingComponent />;
      case 'health':
        return <HealthComponent healthData={healthData} setHealthData={setHealthData} />;
      default:
        return <DreamOComponent />;
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>🎨 AI Image Generation</h1>
        <p>Giao diện thống nhất cho DreamO, OmniGen2 và Training</p>
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

      <main className="content">
        {renderTabContent()}
      </main>
    </div>
  );
}

export default App; 