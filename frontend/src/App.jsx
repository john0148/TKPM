import React, { useState, useRef, useCallback } from 'react';
import { Upload, Image as ImageIcon, Wand2, Plus, Settings, User, Star, Archive, MessageCircle, Bell, ChevronDown, X, Underline } from 'lucide-react';
import { dreamoGenerate, omnigen2InContextGeneration, fileToBase64, validateImageFile, resizeImage } from './utils/api';

const App = () => {
  const [currentPhase, setCurrentPhase] = useState(1);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [selectedImages, setSelectedImages] = useState([]);
  const [zoomImage, setZoomImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [phase1Prompt, setPhase1Prompt] = useState('');
  const [humanImage, setHumanImage] = useState(null);
  const [objectImage, setObjectImage] = useState(null);
  const [errors, setErrors] = useState({});
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDraftMode, setIsDraftMode] = useState(false);
  const [currentGeneratingImage, setCurrentGeneratingImage] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [cursorPosition, setCursorPosition] = useState(0);

  const humanFileRef = useRef(null);
  const objectFileRef = useRef(null);
  const phase2FileRef = useRef(null);

  const handleFileUpload = (file, type) => {
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageData = {
          file,
          url: e.target.result,
          id: Date.now()
        };
        
        if (type === 'human') {
          setHumanImage(imageData);
          setErrors(prev => ({ ...prev, human: null }));
        } else if (type === 'object') {
          setObjectImage(imageData);
          setErrors(prev => ({ ...prev, object: null }));
        } else if (type === 'phase2') {
          setSelectedImages(prev => [...prev, imageData]);
          setErrors(prev => ({ ...prev, images: null }));
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePhase1Generate = async () => {
    setErrors({});
    
    if (!humanImage) {
      setErrors(prev => ({ ...prev, human: 'Please upload a human image' }));
      return;
    }
    if (!objectImage) {
      setErrors(prev => ({ ...prev, object: 'Please upload an object image' }));
      return;
    }
    if (!phase1Prompt.trim()) {
      setErrors(prev => ({ ...prev, phase1Prompt: 'Please enter a prompt' }));
      return;
    }

    setIsGenerating(true);
    
    try {
      // Validate and resize images
      validateImageFile(humanImage.file);
      validateImageFile(objectImage.file);
      
      const resizedHuman = await resizeImage(humanImage.file);
      const resizedObject = await resizeImage(objectImage.file);
      
      // Convert images to base64
      const humanBase64 = await fileToBase64(resizedHuman);
      const objectBase64 = await fileToBase64(resizedObject);
      
      const requestData = {
        prompt: phase1Prompt.trim(),
        ref_images: [
          {
            image_data: humanBase64,
            task: 'id'
          },
          {
            image_data: objectBase64,
            task: 'ip'
          }
        ],
      };
      
      const response = await dreamoGenerate(requestData);
      console.log('Phase 1 generation response:', response);

      if (!response || !response.image) {
        throw new Error('No generated images returned from API');
      }

      // For base64 image data, we need to create the full data URL
      const imageUrl = response.image.startsWith('data:') 
        ? response.image 
        : `data:image/jpeg;base64,${response.image}`;

      const newImage = {
        id: Date.now(),
        url: imageUrl,
        human: humanImage.url,
        object: objectImage.url,
        prompt: phase1Prompt.trim()
      };
      
      setGeneratedImages(prev => [...prev, newImage]);
      
      // Reset all states after successful generation
      setHumanImage(null);
      setObjectImage(null);
      setPhase1Prompt('');
      setErrors({});
      
      // Reset file input values
      if (humanFileRef.current) humanFileRef.current.value = '';
      if (objectFileRef.current) objectFileRef.current.value = '';
      
    } catch (error) {
      console.error('Generation failed:', error);
      setErrors({ general: error.message || 'Generation failed. Please try again.' });
    } finally {
      setIsGenerating(false);
    }
  };

  const handlePhase2Generate = async () => {
    setErrors({});
    
    if (!prompt.trim()) {
      setErrors({ prompt: 'Please enter a prompt' });
      return;
    }
    
    if (selectedImages.length === 0) {
      setErrors({ images: 'Please select at least one image' });
      return;
    }

    setIsGenerating(true);
    setIsDraftMode(true);
    
    // Set placeholder for generating image
    setCurrentGeneratingImage({
      id: Date.now(),
      url: null,
      isGenerating: true,
      prompt: prompt.trim()
    });
    
    try {
      // Convert all selected images to base64
      const imagePromises = selectedImages.map(async (img) => {
        const resizedImage = await resizeImage(img.file);
        return fileToBase64(resizedImage);
      });
      
      const base64Images = await Promise.all(imagePromises);
      console.log('Base64 images for Phase 2:', base64Images);
      
      const omnigen2Data = {
          instruction: prompt.trim(),
          input_images: base64Images,
      }

      const response = await omnigen2InContextGeneration(omnigen2Data);
      console.log('Phase 2 generation response:', response);
      const image_response = response?.image;
      if (!image_response) {
        throw new Error('No generated image returned from API');
      }

      const result_image = response.image.startsWith('data:')
           ? response.image : `data:image/jpeg;base64,${response.image}`;

      const newImage = {
        id: Date.now(),
        url: result_image, 
        isComposed: true,
        prompt: prompt.trim()
      };
      
      setGeneratedImages(prev => [...prev, newImage]);
      setCurrentGeneratingImage(newImage);
      
    } catch (error) {
      console.error('Generation failed:', error);
      setErrors({ general: error.message || 'Generation failed. Please try again.' });
      setIsDraftMode(false);
      setCurrentGeneratingImage(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleBackToGrid = () => {
    setIsDraftMode(false);
    setCurrentGeneratingImage(null);
    setPrompt('');
    setSelectedImages([]);
  };

  const handleDragStart = (e, image) => {
    e.dataTransfer.setData('application/json', JSON.stringify(image));
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    const imageData = JSON.parse(e.dataTransfer.getData('application/json'));
    
    if (!selectedImages.find(img => img.id === imageData.id)) {
      // If the image is from the gallery (already in base64 format)
      if (imageData.url.startsWith('data:')) {
        // Create a mock file object for API consistency
        const base64Response = await fetch(imageData.url);
        const blob = await base64Response.blob();
        const file = new File([blob], `image-${imageData.id}.jpg`, { type: 'image/jpeg' });
        
        const newImageData = {
          ...imageData,
          file: file
        };
        
        setSelectedImages(prev => [...prev, newImageData]);
      } else {
        // If it's a regular file upload
        setSelectedImages(prev => [...prev, imageData]);
      }
      setErrors(prev => ({ ...prev, images: null }));
    }
  };

  const removeSelectedImage = (id) => {
    setSelectedImages(prev => prev.filter(img => img.id !== id));
  };

  const UploadArea = ({ type, image, onClick, error }) => (
    <div
      onClick={onClick}
      className={`relative rounded-xl p-8 cursor-pointer transition-all duration-200 hover:bg-gray-700/50 ${
        error ? 'bg-red-500/10' : 'bg-gray-800/40'
      }`}
      style={{
        background: error 
          ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%)' 
          : 'linear-gradient(135deg, rgba(55, 65, 81, 0.4) 0%, rgba(31, 41, 55, 0.6) 100%)',
        border: '1px solid rgba(75, 85, 99, 0.3)'
      }}
    >
      {image ? (
        <div className="relative">
          <img
            src={image.url}
            alt={type}
            className="w-full h-32 object-contain rounded-lg"
          />
          <div className="absolute inset-0 bg-black/50 rounded-lg flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
            <Plus className="w-6 h-6 text-white" />
          </div>
        </div>
      ) : (
        <div className="text-center">
          <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-200 font-medium">Upload {type}</p>
          <p className="text-gray-400 text-sm mt-1">Click to select image</p>
        </div>
      )}
      {error && (
        <p className="text-red-400 text-sm mt-2 absolute -bottom-6 left-0">{error}</p>
      )}
    </div>
  );

  const getSuggestionsForPhase = useCallback((phase, inputLength = 2) => {
    if (phase === 1) {
      return ['ref#1', 'ref#2'];
    } else {
      return Array.from({ length: inputLength }, (_, i) => `|image_${i + 1}|`);
    }
  }, []);

  const handlePromptKeyDown = (e, type) => {
    if (e.key === '@') {
      e.preventDefault(); // Prevent the @ from being added to the input
      setShowSuggestions(true);
      setSuggestions(type === 'phase1' 
        ? getSuggestionsForPhase(1)
        : getSuggestionsForPhase(2, selectedImages.length)
      );
      setCursorPosition(e.target.selectionStart);
    }
  };

  const handleSuggestionClick = (suggestion, type) => {
    if (type === 'phase1') {
      const beforeCursor = phase1Prompt.substring(0, cursorPosition);
      const afterCursor = phase1Prompt.substring(cursorPosition);
      setPhase1Prompt(beforeCursor + suggestion + ' ' + afterCursor);
    } else {
      const beforeCursor = prompt.substring(0, cursorPosition);
      const afterCursor = prompt.substring(cursorPosition);
      setPrompt(beforeCursor + suggestion + ' ' + afterCursor);
    }
    setShowSuggestions(false);
  };

  const SuggestionBox = ({ suggestions, onSelect, type }) => (
    <div 
      className={`absolute z-10 w-48 rounded-md shadow-lg bg-gray-800 ring-1 ring-black ring-opacity-5 ${
        type === 'phase2' ? 'bottom-full mb-2' : 'mt-2'
      }`}
      style={{
        border: '1px solid rgba(75, 85, 99, 0.3)'
      }}
    >
      <ul className="py-1">
        {suggestions.map((suggestion, index) => (
          <li
            key={index}
            onClick={() => onSelect(suggestion, type)}
            className="px-4 py-2 text-sm text-gray-200 hover:bg-gray-700 cursor-pointer"
          >
            {suggestion}
          </li>
        ))}
      </ul>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white" style={{ backgroundColor: '#0f0f0f' }}>
      <div className="flex h-screen">
        {/* Left Sidebar */}
        <div 
          className="w-64 backdrop-blur-sm border-r flex flex-col"
          style={{
            backgroundColor: 'rgba(20, 20, 20, 0.8)',
            borderColor: 'rgba(55, 65, 81, 0.3)'
          }}
        >
          {/* Logo/Brand */}
          <div className="p-4 border-b" style={{ borderColor: 'rgba(55, 65, 81, 0.3)' }}>
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)'
              }}>
                <Wand2 className="w-5 h-5 text-white" />
              </div>
              <span className="font-semibold text-lg">ImageGen</span>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex-1 p-4">
            <nav className="space-y-2">
              <button 
                className="w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-white"
                style={{
                  background: 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)'
                }}
              >
                <Wand2 className="w-5 h-5" />
                <span>Generate</span>
              </button>
              <button className="w-full flex items-center space-x-3 px-3 py-2 text-gray-300 hover:text-white hover:bg-gray-700/30 rounded-lg transition-colors">
                <ImageIcon className="w-5 h-5" />
                <span>My Images</span>
              </button>
              <button className="w-full flex items-center space-x-3 px-3 py-2 text-gray-300 hover:text-white hover:bg-gray-700/30 rounded-lg transition-colors">
                <Star className="w-5 h-5" />
                <span>Favorites</span>
              </button>
              <button className="w-full flex items-center space-x-3 px-3 py-2 text-gray-300 hover:text-white hover:bg-gray-700/30 rounded-lg transition-colors">
                <Archive className="w-5 h-5" />
                <span>Archive</span>
              </button>
            </nav>

            <div className="mt-8">
              <h3 className="text-gray-400 text-sm font-medium mb-3">Gallery</h3>
              <div className="space-y-3">
                {generatedImages.map((image) => (
                  <div
                    key={image.id}
                    draggable={currentPhase === 2 && !isDraftMode}
                    onDragStart={(e) => handleDragStart(e, image)}
                    onClick={() => setZoomImage(image.url)}
                    className={`relative group cursor-pointer rounded-lg overflow-hidden ${
                      currentPhase === 2 && !isDraftMode ? 'hover:scale-105 transition-transform' : ''
                    }`}
                  >
                    <img
                      src={image.url}
                      alt="Generated"
                      className="w-full h-16 object-cover rounded-lg"
                    />
                    {image.isComposed && (
                      <div className="absolute top-1 right-1 bg-green-500 text-white text-xs px-1 py-0.5 rounded">
                        Composed
                      </div>
                    )}
                    <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                      <p className="text-white text-xs">
                        {currentPhase === 2 && !isDraftMode ? 'Drag to add' : 'Click to show'}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* User Profile */}
          <div className="p-4 border-t" style={{ borderColor: 'rgba(55, 65, 81, 0.3)' }}>
            <div className="flex items-center space-x-3">
              <div 
                className="w-8 h-8 rounded-full flex items-center justify-center"
                style={{
                  background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)'
                }}
              >
                <User className="w-4 h-4 text-white" />
              </div>
              <div className="flex-1">
                <p className="text-sm font-medium">User</p>
                <p className="text-xs text-gray-400">Free Plan</p>
              </div>
              <Settings className="w-4 h-4 text-gray-400" />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Header with Phase Switcher */}
          <div 
            className="backdrop-blur-sm border-b p-4"
            style={{
              backgroundColor: 'rgba(20, 20, 20, 0.6)',
              borderColor: 'rgba(55, 65, 81, 0.3)'
            }}
          >
            <div className="flex justify-center">
              <div 
                className="flex backdrop-blur-sm rounded-xl p-1"
                style={{
                  backgroundColor: 'rgba(31, 41, 55, 0.5)'
                }}
              >
                <button
                  onClick={() => {
                    setCurrentPhase(1);
                    // Only reset draft mode if we're not currently generating
                    if (!isGenerating) {
                      setIsDraftMode(false);
                      setCurrentGeneratingImage(null);
                    }
                  }}
                  className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 ${
                    currentPhase === 1
                      ? 'text-white shadow-lg'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700/30'
                  }`}
                  style={{
                    background: currentPhase === 1 ? 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)' : 'transparent'
                  }}
                >
                  Human w/ Object
                </button>
                <button
                  onClick={() => {
                    setCurrentPhase(2);
                    // Only reset draft mode if we're not currently generating
                    if (!isGenerating) {
                      setIsDraftMode(false);
                      setCurrentGeneratingImage(null);
                    }
                  }}
                  className={`px-6 py-2 rounded-lg font-medium transition-all duration-200 ${
                    currentPhase === 2
                      ? 'text-white shadow-lg'
                      : 'text-gray-300 hover:text-white hover:bg-gray-700/30'
                  }`}
                  style={{
                    background: currentPhase === 2 ? 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)' : 'transparent'
                  }}
                >
                  Multiple Object
                </button>
              </div>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 flex flex-col">
            <div className="flex-1 p-8">
              {currentPhase === 1 ? (
                /* Phase 1 Content */
                <div className="max-w-4xl mx-auto h-full flex flex-col">
                  <div className="flex-1 flex items-center justify-center">
                    <div className="w-full max-w-2xl">
                      <div className="mb-8">
                        <div className="relative">
                          <input
                            type="text"
                            value={phase1Prompt}
                            onChange={(e) => {
                              setPhase1Prompt(e.target.value);
                              setErrors(prev => ({ ...prev, phase1Prompt: null }));
                            }}
                            onKeyDown={(e) => handlePromptKeyDown(e, 'phase1')}
                            placeholder="Enter your prompt (use @ for reference suggestions)..."
                            className={`w-full backdrop-blur-sm rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all ${
                              errors.phase1Prompt ? 'ring-2 ring-red-500' : ''
                            }`}
                            style={{
                              backgroundColor: 'rgba(31, 41, 55, 0.5)',
                              border: '1px solid rgba(75, 85, 99, 0.3)'
                            }}
                          />
                          {showSuggestions && (
                            <SuggestionBox
                              suggestions={suggestions}
                              onSelect={handleSuggestionClick}
                              type="phase1"
                            />
                          )}
                        </div>
                        {errors.phase1Prompt && (
                          <p className="text-red-400 text-sm mt-1">{errors.phase1Prompt}</p>
                        )}
                      </div>
                      <div className="grid grid-cols-2 gap-8">
                        <div className="relative">
                          <UploadArea
                            type="Human"
                            image={humanImage}
                            onClick={() => humanFileRef.current?.click()}
                            error={errors.human}
                          />
                          <input
                            ref={humanFileRef}
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileUpload(e.target.files[0], 'human')}
                            className="hidden"
                          />
                        </div>
                        <div className="relative">
                          <UploadArea
                            type="Object"
                            image={objectImage}
                            onClick={() => objectFileRef.current?.click()}
                            error={errors.object}
                          />
                          <input
                            ref={objectFileRef}
                            type="file"
                            accept="image/*"
                            onChange={(e) => handleFileUpload(e.target.files[0], 'object')}
                            className="hidden"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                  {errors.general && (
                    <p className="text-red-400 text-sm text-center mb-4">{errors.general}</p>
                  )}
                </div>
              ) : (
                /* Phase 2 Content */
                <div className="max-w-6xl mx-auto h-full flex flex-col">
                  {!isDraftMode ? (
                    /* Grid Mode */
                    <div className="flex-1">
                      <div
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        className={`rounded-xl p-8 h-full min-h-96 transition-all duration-200 ${
                          errors.images ? 'bg-red-500/10' : ''
                        }`}
                        style={{
                          background: errors.images 
                            ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%)' 
                            : 'linear-gradient(135deg, rgba(55, 65, 81, 0.3) 0%, rgba(31, 41, 55, 0.5) 100%)',
                          border: '1px solid rgba(75, 85, 99, 0.3)'
                        }}
                      >
                        <h3 className="text-lg font-medium mb-6 text-center">Selected Images</h3>
                        {selectedImages.length === 0 ? (
                          <div className="text-center text-gray-400 h-full flex flex-col items-center justify-center">
                            <ImageIcon className="w-16 h-16 mx-auto mb-4" />
                            <p className="text-lg">Drag images from gallery or upload to compose</p>
                            <p className="text-sm mt-2">Select images from the sidebar and drop them here</p>
                          </div>
                        ) : (
                          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
                            {selectedImages.map((image) => (
                              <div key={image.id} className="relative group">
                                <img
                                  src={image.url}
                                  alt="Selected"
                                  className="w-full h-40 object-cover rounded-lg cursor-zoom-in"
                                  style={{ backgroundColor: 'rgba(31, 41, 55, 0.5)' }}
                                  onClick={() => setZoomImage(image.url)}
                                />
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    removeSelectedImage(image.id);
                                  }}
                                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm opacity-0 group-hover:opacity-100 transition-opacity z-10"
                                >
                                  <X className="w-4 h-4" />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                      {errors.images && (
                        <p className="text-red-400 text-sm mt-2 text-center">{errors.images}</p>
                      )}
                    </div>
                  ) : (
                    /* Draft Mode */
                    <div className="flex-1 flex gap-6">
                      {/* Large Generation Panel */}
                      <div className="flex-1">
                        <div
                          className="rounded-xl p-8 h-full flex items-center justify-center"
                          style={{
                            background: 'linear-gradient(135deg, rgba(55, 65, 81, 0.3) 0%, rgba(31, 41, 55, 0.5) 100%)',
                            border: '1px solid rgba(75, 85, 99, 0.3)'
                          }}
                        >                                  {currentGeneratingImage ? (
                            <div className="text-center">
                              {isGenerating ? (
                                <div className="animate-pulse">
                                  <div className="w-64 h-64 bg-gray-600 rounded-lg mx-auto mb-4"></div>
                                  <p className="text-gray-300">Generating image...</p>
                                  <p className="text-sm text-gray-400 mt-2">"{currentGeneratingImage.prompt}"</p>
                                </div>
                              ) : (
                                <div>
                                  <img
                                    src={currentGeneratingImage.url}
                                    alt="Generated"
                                    className="w-64 h-64 object-cover rounded-lg mx-auto mb-4 cursor-zoom-in"
                                    onClick={() => setZoomImage(currentGeneratingImage.url)}
                                  />
                                  <p className="text-gray-300">Generation complete!</p>
                                  <p className="text-sm text-gray-400 mt-2">"{currentGeneratingImage.prompt}"</p>
                                </div>
                              )}
                            </div>
                          ) : (
                            <div className="text-center text-gray-400">
                              <Wand2 className="w-16 h-16 mx-auto mb-4" />
                              <p>Generated image will appear here</p>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Right Sidebar - Uploaded Images */}
                      <div className="w-80">
                        <div
                          className="rounded-xl p-4 h-full"
                          style={{
                            background: 'linear-gradient(135deg, rgba(55, 65, 81, 0.3) 0%, rgba(31, 41, 55, 0.5) 100%)',
                            border: '1px solid rgba(75, 85, 99, 0.3)'
                          }}
                        >
                          <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-medium">Source Images</h3>
                            <button
                              onClick={handleBackToGrid}
                              className="text-gray-400 hover:text-white transition-colors"
                            >
                              <X className="w-5 h-5" />
                            </button>
                          </div>
                          <div className="space-y-3 max-h-full overflow-y-auto">
                            {selectedImages.map((image) => (
                              <div key={image.id} className="relative group" onClick={() => setZoomImage(image.url)}>
                                <img
                                  src={image.url}
                                  alt="Source"
                                  className="w-full h-24 object-cover rounded-lg cursor-zoom-in"
                                />
                                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg" />
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                  {errors.general && (
                    <p className="text-red-400 text-sm text-center mb-4">{errors.general}</p>
                  )}
                </div>
              )}
            </div>

            {/* Bottom Controls */}
            <div 
              className="backdrop-blur-sm border-t p-6"
              style={{
                backgroundColor: 'rgba(20, 20, 20, 0.6)',
                borderColor: 'rgba(55, 65, 81, 0.3)'
              }}
            >
              <div className="max-w-6xl mx-auto">
                {currentPhase === 1 ? (
                  <div className="flex justify-center">
                    <button
                      onClick={handlePhase1Generate}
                      disabled={isGenerating}
                      className="text-white px-8 py-3 rounded-xl font-medium transition-all duration-200 flex items-center shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        background: isGenerating 
                          ? 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
                          : 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)'
                      }}
                    >
                      <Wand2 className="w-5 h-5 mr-2" />
                      {isGenerating ? 'Generating...' : 'Generate'}
                    </button>
                  </div>
                ) : !isDraftMode ? (
                  <div className="flex gap-4">
                    <div className="flex-1">
                      <div className="relative">
                        <input
                          type="text"
                          value={prompt}
                          onChange={(e) => {
                            setPrompt(e.target.value);
                            setErrors(prev => ({ ...prev, prompt: null }));
                          }}
                          onKeyDown={(e) => handlePromptKeyDown(e, 'phase2')}
                          placeholder="Describe the image (use @ for reference suggestions)..."
                          className={`w-full backdrop-blur-sm rounded-xl px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all ${
                            errors.prompt ? 'ring-2 ring-red-500' : ''
                          }`}
                          style={{
                            backgroundColor: 'rgba(31, 41, 55, 0.5)',
                            border: '1px solid rgba(75, 85, 99, 0.3)'
                          }}
                        />
                        {showSuggestions && (
                          <SuggestionBox
                            suggestions={suggestions}
                            onSelect={handleSuggestionClick}
                            type="phase2"
                          />
                        )}
                      </div>
                      {errors.prompt && (
                        <p className="text-red-400 text-sm mt-1">{errors.prompt}</p>
                      )}
                    </div>
                    <button
                      onClick={handlePhase2Generate}
                      disabled={isGenerating}
                      className="text-white px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{
                        background: isGenerating 
                          ? 'linear-gradient(135deg, #6b7280 0%, #4b5563 100%)'
                          : 'linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%)'
                      }}
                    >
                      <Wand2 className="w-5 h-5 mr-2" />
                      {isGenerating ? 'Generating...' : 'Generate'}
                    </button>
                    <button
                      className="text-white px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center backdrop-blur-sm hover:bg-gray-600/30"
                      onClick={() => phase2FileRef.current?.click()}
                      style={{
                        backgroundColor: 'rgba(55, 65, 81, 0.5)',
                        border: '1px solid rgba(75, 85, 99, 0.3)'
                      }}
                    >
                      <Upload className="w-5 h-5 mr-2" />
                      Upload
                    </button>
                    <input
                      ref={phase2FileRef}
                      type="file"
                      accept="image/*"
                      onChange={(e) => handleFileUpload(e.target.files[0], 'phase2')}
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="flex justify-center">
                    <button
                      onClick={handleBackToGrid}
                      className="text-white px-8 py-3 rounded-xl font-medium transition-all duration-200 flex items-center backdrop-blur-sm hover:bg-gray-600/30"
                      style={{
                        backgroundColor: 'rgba(55, 65, 81, 0.5)',
                        border: '1px solid rgba(75, 85, 99, 0.3)'
                      }}
                    >
                      Back to Grid
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Zoom Modal */}
      {zoomImage && (
        <div 
          className="fixed inset-0 z-50 flex items-center justify-center transition-all will-change-[backdrop-filter,background-color]"
          onClick={() => setZoomImage(null)}
          style={{
            animation: 'fadeIn 0.2s ease-out forwards',
            WebkitBackfaceVisibility: 'hidden',
            backfaceVisibility: 'hidden'
          }}
        >
          <style>
            {`
              @keyframes fadeIn {
                0% { background-color: rgba(0, 0, 0, 0); backdrop-filter: blur(0px); }
                100% { background-color: rgba(0, 0, 0, 0.85); backdrop-filter: blur(12px); }
              }
              @keyframes zoomIn {
                0% { transform: scale(0.9); opacity: 0; filter: blur(8px); }
                100% { transform: scale(1); opacity: 1; filter: blur(0); }
              }
            `}
          </style>
          <div 
            className="relative max-w-[90vw] max-h-[90vh] will-change-transform"
            style={{
              animation: 'zoomIn 0.25s cubic-bezier(0.16, 1, 0.3, 1) forwards',
              WebkitBackfaceVisibility: 'hidden',
              backfaceVisibility: 'hidden'
            }}
          >
            <img
              src={zoomImage}
              alt="Zoomed"
              className="max-w-full max-h-[90vh] object-contain rounded-lg shadow-2xl"
            />
            <button
              onClick={(e) => {
                e.stopPropagation();
                setZoomImage(null);
              }}
              className="absolute -top-4 -right-4 bg-white/10 hover:bg-white/20 text-white rounded-full p-2 backdrop-blur-sm transition-all duration-200 hover:scale-110"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;