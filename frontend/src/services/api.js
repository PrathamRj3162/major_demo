/**
 * API Service Layer
 * ==================
 * Centralized API communication module.
 * All backend API calls go through this service.
 */

import axios from 'axios';

// In production, VITE_API_URL points to the deployed backend (e.g. https://xxx.onrender.com/api)
// In local development, Vite proxy forwards /api to localhost:5001
const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
    baseURL: API_BASE,
    timeout: 300000, // 5 min timeout for federated training
});

/**
 * Upload an X-ray image and get a prediction with Grad-CAM.
 * @param {File} file - The image file to upload.
 * @returns {Promise} API response with prediction, confidence, and Grad-CAM.
 */
export const predictImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

/**
 * Upload an X-ray image only (without prediction).
 * @param {File} file - The image file to upload.
 * @returns {Promise} API response with filename.
 */
export const uploadImage = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

/**
 * Generate Grad-CAM for an uploaded image.
 * @param {File} file - The image file.
 * @returns {Promise} API response with Grad-CAM images.
 */
export const generateGradCAM = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/gradcam', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

/**
 * Run federated learning simulation.
 * @param {number} numClients - Number of simulated clients.
 * @param {number} numRounds - Number of training rounds.
 * @returns {Promise} API response with training logs.
 */
export const runFederatedTraining = async (numClients = 4, numRounds = 5) => {
    const response = await api.post('/federated-train', {
        num_clients: numClients,
        num_rounds: numRounds,
    });
    return response.data;
};

/**
 * Get model performance statistics.
 * @returns {Promise} API response with metrics, confusion matrix, etc.
 */
export const getModelStats = async () => {
    const response = await api.get('/model-stats');
    return response.data;
};

/**
 * Health check.
 * @returns {Promise} API health status.
 */
export const healthCheck = async () => {
    const response = await api.get('/health');
    return response.data;
};

export default api;
