import React, { useState } from 'react';
import './ImageGallery.css';

const ImageGallery = ({ images, onDeleteImage }) => {
  const [selectedImage, setSelectedImage] = useState(null);

  const handleImageClick = (image) => {
    setSelectedImage(image);
  };

  const handleCloseModal = () => {
    setSelectedImage(null);
  };

  const handleDeleteImage = (imageId, e) => {
    e.stopPropagation();
    onDeleteImage(imageId);
  };

  return (
    <div className="image-gallery">
      <div className="gallery-header">
        <h3>Generated Images</h3>
        <span className="image-count">{images.length} images</span>
      </div>
      
      <div className="gallery-grid">
        {images.map((image) => (
          <div 
            key={image.id} 
            className="gallery-item"
            onClick={() => handleImageClick(image)}
          >
            <img 
              src={image.url} 
              alt={image.prompt || 'Generated image'} 
              className="gallery-thumbnail"
            />
            <div className="gallery-item-overlay">
              <button 
                className="delete-btn"
                onClick={(e) => handleDeleteImage(image.id, e)}
                title="Delete image"
              >
                ×
              </button>
            </div>
            {image.prompt && (
              <div className="gallery-item-prompt">
                {image.prompt.length > 50 
                  ? `${image.prompt.substring(0, 50)}...` 
                  : image.prompt
                }
              </div>
            )}
          </div>
        ))}
      </div>

      {images.length === 0 && (
        <div className="empty-gallery">
          <div className="empty-icon">🖼️</div>
          <p>No generated images yet</p>
          <small>Generated images will appear here</small>
        </div>
      )}

      {/* Modal for full-size image view */}
      {selectedImage && (
        <div className="image-modal" onClick={handleCloseModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={handleCloseModal}>×</button>
            <img 
              src={selectedImage.url} 
              alt={selectedImage.prompt || 'Generated image'} 
              className="modal-image"
            />
            {selectedImage.prompt && (
              <div className="modal-prompt">
                <h4>Prompt:</h4>
                <p>{selectedImage.prompt}</p>
              </div>
            )}
            <div className="modal-info">
              <span>Generated: {new Date(selectedImage.timestamp).toLocaleString()}</span>
              <span>Model: {selectedImage.model}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageGallery; 