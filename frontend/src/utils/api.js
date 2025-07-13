const API_HOST_URL = 'https://f2bca0cff3dd.ngrok-free.app'; 
const API_BASE_URL = 'api';
export const API_URL = `${API_HOST_URL}/${API_BASE_URL}`;


// Helper function để convert file thành base64
export const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
};

// Helper function để thực hiện API calls
const apiCall = async (endpoint, options = {}) => {
  try {
    console.log(`Calling API: ${API_URL}${endpoint}`, options);
    const response = await fetch(`${API_URL}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
};

// Helper function cho multipart form data
const apiFormCall = async (endpoint, formData) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('API form call failed:', error);
    throw error;
  }
};

// =================
// HEALTH CHECK APIs
// =================

export const healthCheck = async () => {
  return apiCall('/health');
};

export const dreamoHealthCheck = async () => {
  return apiCall('/dreamo/health');
};

export const omnigen2HealthCheck = async () => {
  return apiCall('/omnigen2/health');
};

// =================
// DREAMO APIs
// =================

export const dreamoGenerate = async (params) => {
  return apiCall('/dreamo/generate', {
    method: 'POST',
    body: JSON.stringify(params)
  });
};

export const getDreamoExamples = async () => {
  return apiCall('/dreamo/examples');
};

export const getDreamoModelInfo = async () => {
  return apiCall('/dreamo/models/info');
};

// =================
// OMNIGEN2 APIs
// =================

export const omnigen2InContextGeneration = async (params) => {
  return apiCall('/omnigen2/in-context-generation', {
    method: 'POST',
    body: JSON.stringify(params)
  });
};

export const omnigen2EditImage = async (params) => {
  return apiCall('/omnigen2/edit', {
    method: 'POST',
    body: JSON.stringify(params)
  });
};

export const getOmnigen2Examples = async () => {
  return apiCall('/omnigen2/examples');
};

export const getOmnigen2ModelInfo = async () => {
  return apiCall('/omnigen2/models/info');
};

// =================
// TRAINING APIs
// =================

export const startTraining = async (nameObject, description, referenceImages) => {
  const formData = new FormData();
  formData.append('name_object', nameObject);
  if (description) {
    formData.append('description', description);
  }
  
  referenceImages.forEach((file, index) => {
    formData.append('reference_images', file);
  });

  return apiFormCall('/training/start', formData);
};

export const getTrainingStatus = async (trainingId) => {
  return apiCall(`/training/status/${trainingId}`);
};

export const inferenceWithTrainedModel = async (trainingId, params) => {
  return apiCall(`/training/inference/${trainingId}`, {
    method: 'POST',
    body: JSON.stringify(params)
  });
};

export const listTrainingSessions = async () => {
  return apiCall('/training/list');
};

export const deleteTrainingSession = async (trainingId) => {
  return apiCall(`/training/delete/${trainingId}`, {
    method: 'DELETE'
  });
};

// =================
// UTILITY FUNCTIONS
// =================

export const validateImageFile = (file) => {
  const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const maxSize = 10 * 1024 * 1024; // 10MB

  if (!validTypes.includes(file.type)) {
    throw new Error('File phải là định dạng JPEG, PNG hoặc WebP');
  }

  if (file.size > maxSize) {
    throw new Error('File không được vượt quá 10MB');
  }

  return true;
};

export const resizeImage = (file, maxWidth = 2048, maxHeight = 2048, quality = 0.9) => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      let { width, height } = img;

      // Calculate new dimensions
      if (width > maxWidth || height > maxHeight) {
        const ratio = Math.min(maxWidth / width, maxHeight / height);
        width *= ratio;
        height *= ratio;
      }

      canvas.width = width;
      canvas.height = height;

      // Draw và compress
      ctx.drawImage(img, 0, 0, width, height);
      
      canvas.toBlob(resolve, file.type, quality);
    };

    img.src = URL.createObjectURL(file);
  });
}; 