// Upload page — drag-and-drop an X-ray and hit "Analyze with AI"

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Upload, Scan, FileWarning, Info } from 'lucide-react';
import ImageUploader from '../components/ImageUploader';
import LoadingSpinner from '../components/LoadingSpinner';
import { predictImage } from '../services/api';

export default function UploadPage() {
    const navigate = useNavigate();
    const [selectedFile, setSelectedFile] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [error, setError] = useState(null);

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setIsAnalyzing(true);
        setError(null);

        try {
            const result = await predictImage(selectedFile);
            navigate('/results', { state: { result } });
        } catch (err) {
            console.error('Analysis failed:', err);
            setError(
                err.response?.data?.error ||
                'Analysis failed. Please check if the backend server is running.'
            );
            setIsAnalyzing(false);
        }
    };

    return (
        <div className="max-w-3xl mx-auto space-y-6">
            <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
            >
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                    <Upload className="w-8 h-8 text-primary-400" />
                    Upload Chest X-Ray
                </h1>
                <p className="text-dark-400">
                    Upload a chest X-ray image for AI-powered pneumonia detection with explainable results.
                </p>
            </motion.div>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="glass-card p-6"
            >
                <ImageUploader
                    onFileSelect={setSelectedFile}
                    isLoading={isAnalyzing}
                />

                {selectedFile && !isAnalyzing && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="mt-6 flex justify-center"
                    >
                        <button
                            onClick={handleAnalyze}
                            className="btn-primary flex items-center gap-3 text-lg px-8 py-4"
                        >
                            <Scan className="w-5 h-5" />
                            Analyze with AI
                        </button>
                    </motion.div>
                )}

                {isAnalyzing && (
                    <div className="mt-6">
                        <LoadingSpinner message="Running AI Analysis (this may take a moment)..." />
                    </div>
                )}
            </motion.div>

            {error && (
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-card p-5 border-red-800/30 bg-red-900/10"
                >
                    <div className="flex items-start gap-3">
                        <FileWarning className="w-5 h-5 text-red-400 mt-0.5 flex-shrink-0" />
                        <div>
                            <h3 className="text-red-300 font-semibold mb-1">Analysis Failed</h3>
                            <p className="text-red-400/80 text-sm">{error}</p>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* how it works section */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="glass-card p-5"
            >
                <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-accent-400 mt-0.5 flex-shrink-0" />
                    <div>
                        <h3 className="text-white font-semibold mb-2">How it works</h3>
                        <ol className="text-dark-400 text-sm space-y-2">
                            <li className="flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-primary-600/20 text-primary-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">1</span>
                                <span>Upload a chest X-ray image (PNG, JPG, JPEG)</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-primary-600/20 text-primary-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">2</span>
                                <span>Image is preprocessed (resize to 224×224, normalize with ImageNet stats)</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-primary-600/20 text-primary-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">3</span>
                                <span>DenseNet121 model runs inference and outputs confidence scores</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-primary-600/20 text-primary-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">4</span>
                                <span>Grad-CAM generates visual explanation highlighting key regions</span>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="w-5 h-5 rounded-full bg-primary-600/20 text-primary-400 text-xs flex items-center justify-center flex-shrink-0 mt-0.5">5</span>
                                <span>If confidence is below 70%, the result is flagged as "Needs Review"</span>
                            </li>
                        </ol>
                    </div>
                </div>
            </motion.div>
        </div>
    );
}
