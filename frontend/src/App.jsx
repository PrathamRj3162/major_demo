/**
 * App.jsx — Main Application Shell
 * ==================================
 * Sets up React Router and the sidebar layout.
 * All pages are rendered inside the main content area.
 */

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import UploadPage from './pages/UploadPage';
import ResultsPage from './pages/ResultsPage';
import FederatedPage from './pages/FederatedPage';
import PerformancePage from './pages/PerformancePage';

export default function App() {
    return (
        <Router>
            <div className="app-layout">
                <Sidebar />
                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<Dashboard />} />
                        <Route path="/upload" element={<UploadPage />} />
                        <Route path="/results" element={<ResultsPage />} />
                        <Route path="/federated" element={<FederatedPage />} />
                        <Route path="/performance" element={<PerformancePage />} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}
