# 🏥 AI-Driven Precision Diagnostics System

## Pneumonia Detection using Deep Learning, Explainable AI & Federated Learning

> A production-level medical AI system that detects pneumonia from chest X-ray images using DenseNet121 (Transfer Learning), provides visual explanations via Grad-CAM, and demonstrates privacy-preserving federated learning across simulated hospital nodes.

---

## 🎯 Project Overview

| Feature | Technology |
|---------|-----------|
| **Deep Learning Model** | DenseNet121 (Transfer Learning from ImageNet) |
| **Explainable AI** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **Federated Learning** | FedAvg algorithm across 4 simulated hospital clients |
| **Backend API** | Flask + PyTorch |
| **Frontend UI** | React 18 + Vite + Tailwind CSS |
| **Visualization** | Recharts, Custom SVG gauges, Framer Motion animations |

---

## 📁 Project Structure

```
btech_major_final_project/
├── backend/
│   ├── app.py                  # Flask application entry point
│   ├── config.py               # Configuration constants
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   ├── densenet_model.py   # DenseNet121 model architecture
│   │   └── model_loader.py     # Singleton model loader
│   ├── services/
│   │   ├── prediction.py       # Inference + confidence scoring
│   │   ├── gradcam.py          # Grad-CAM heatmap generation
│   │   └── federated.py        # FedAvg simulation engine
│   ├── routes/
│   │   ├── predict_routes.py   # /api/predict, /api/upload
│   │   ├── gradcam_routes.py   # /api/gradcam
│   │   ├── federated_routes.py # /api/federated-train
│   │   └── stats_routes.py     # /api/model-stats
│   ├── utils/
│   │   ├── preprocessing.py    # Image resize, normalize pipeline
│   │   └── helpers.py          # Base64 encoding, file helpers
│   └── uploads/                # Uploaded X-ray images
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Sidebar.jsx         # Navigation sidebar
│   │   │   ├── StatCard.jsx        # Animated stat cards
│   │   │   ├── ConfidenceGauge.jsx # SVG circular gauge
│   │   │   ├── ImageUploader.jsx   # Drag-and-drop upload
│   │   │   ├── GradCamViewer.jsx   # Grad-CAM visualization
│   │   │   └── LoadingSpinner.jsx  # Loading animation
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx       # Home/overview page
│   │   │   ├── UploadPage.jsx      # Upload X-ray page
│   │   │   ├── ResultsPage.jsx     # Analysis results
│   │   │   ├── FederatedPage.jsx   # Federated learning dashboard
│   │   │   └── PerformancePage.jsx # Model metrics & charts
│   │   ├── services/
│   │   │   └── api.js              # API service layer (axios)
│   │   ├── App.jsx                 # Main app with routing
│   │   ├── main.jsx                # Entry point
│   │   └── index.css               # Tailwind + custom styles
│   ├── package.json
│   ├── vite.config.js              # Vite config with API proxy
│   ├── tailwind.config.js          # Custom theme & animations
│   └── index.html
├── data/                           # Sample dataset guide
└── README.md
```

---

## 🚀 Setup & Installation

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm
- **Git**

### Step 1: Clone the Repository
```bash
cd /path/to/btech_major_final_project
```

### Step 2: Backend Setup
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Mac/Linux
# .\venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```
The backend will start at `http://localhost:5000`.

### Step 3: Frontend Setup
```bash
# Open a new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```
The frontend will start at `http://localhost:5173`.

### Step 4: Open in Browser
Navigate to `http://localhost:5173` in your browser.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/upload` | Upload an X-ray image |
| `POST` | `/api/predict` | Run inference (upload + predict + Grad-CAM in one call) |
| `POST` | `/api/gradcam` | Generate Grad-CAM visualization only |
| `POST` | `/api/federated-train` | Run federated learning simulation |
| `GET` | `/api/model-stats` | Get model performance metrics |

---

## 🧠 AI Pipeline — How It Works

```
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
  │  Upload X-Ray │────▶│  Preprocessing   │────▶│  DenseNet121      │
  │  (PNG/JPG)    │     │  Resize 224×224  │     │  Forward Pass     │
  │               │     │  RGB Convert     │     │  Softmax → P(cls) │
  └──────────────┘     │  ImageNet Norm   │     └────────┬─────────┘
                        └─────────────────┘              │
                                                         ▼
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
  │  Final Result │◀────│ Decision Logic   │◀────│   Grad-CAM        │
  │  + Grad-CAM   │     │ conf < 0.7 ?     │     │   Heatmap Gen     │
  │  + Review Flag│     │ → Needs Review   │     │   Overlay on img  │
  └──────────────┘     └─────────────────┘     └──────────────────┘
```

1. **Image Upload**: User uploads a chest X-ray (PNG/JPEG)
2. **Preprocessing**: Resize to 224×224, convert to RGB, normalize with ImageNet mean/std
3. **Model Inference**: DenseNet121 outputs logits → softmax for probabilities
4. **Confidence Decision**: If max probability < 70% → flagged as "Needs Review"
5. **Grad-CAM**: Hooks on the last DenseBlock capture activations + gradients → heatmap overlay
6. **Response**: Prediction, confidence score, and Grad-CAM visualization returned

---

## 🔗 Federated Learning — FedAvg Algorithm

```
  ┌───────────────── Central Server ─────────────────┐
  │   θ_global = Σ (n_k / n) × θ_k                   │
  │   (Weighted average of all client parameters)      │
  └──────┬────────┬────────┬────────┬────────────────┘
         │        │        │        │
   ┌─────┴──┐ ┌──┴─────┐ ┌┴──────┐ ┌┴──────┐
   │ Hosp A │ │ Hosp B │ │Hosp C │ │Hosp D │
   │ n=80   │ │ n=75   │ │ n=85  │ │ n=70  │
   └────────┘ └────────┘ └───────┘ └───────┘
```

1. Server distributes global model to all clients
2. Each client trains locally on its private data shard
3. Clients send updated model weights (NOT data) to server
4. Server aggregates via FedAvg: weighted average by dataset size
5. Repeat for N rounds until convergence

---

## 🖼️ UI Pages

1. **Dashboard** — System overview, stats, quick actions, model info
2. **Upload X-Ray** — Drag-and-drop with preview, one-click analysis
3. **Results** — Prediction label, confidence gauge, Grad-CAM overlay, probability bars
4. **Federated Learning** — Configurable simulation, convergence charts, client comparison
5. **Model Performance** — Confusion matrix, loss/accuracy curves, ROC curve, classification report

---

## 📊 Sample Dataset

This project is designed to work with the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:

- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: ~2GB (5,863 images)
- **Classes**: NORMAL, PNEUMONIA
- **Split**: train/val/test

To use with a trained model:
1. Download the dataset from Kaggle
2. Train DenseNet121 using the training scripts
3. Save the best model weights to `backend/models/best_model.pth`

> **Note**: The system works out-of-the-box with ImageNet pretrained weights (no fine-tuned model needed). Fine-tuning on the chest X-ray dataset improves accuracy from ~50% to ~95%+.

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Model** | PyTorch + torchvision | DenseNet121 transfer learning |
| **XAI** | Custom Grad-CAM | Visual explanations |
| **FL** | Custom FedAvg | Federated learning simulation |
| **Backend** | Flask + Flask-CORS | REST API server |
| **Frontend** | React 18 + Vite | UI framework |
| **Styling** | Tailwind CSS 3 | Utility-first CSS |
| **Charts** | Recharts | Data visualization |
| **Animation** | Framer Motion | UI animations |
| **Icons** | Lucide React | Icon library |
| **HTTP** | Axios | API communication |
| **Image** | OpenCV + Pillow | Image processing |

---

## 🎓 Key Concepts for Viva

1. **Transfer Learning**: Reusing ImageNet-pretrained DenseNet121 features instead of training from scratch
2. **Grad-CAM**: Uses gradients flowing into the last convolutional layer to produce a coarse localization map highlighting important regions
3. **FedAvg**: Federated Averaging — each client trains locally, server aggregates by weighted average of parameters
4. **Confidence Thresholding**: Low-confidence predictions are flagged for human review, improving clinical safety
5. **DenseNet Architecture**: Dense connections (feature reuse) reduce parameters while maintaining performance

---

## 📄 License

This project is for educational purposes (B.Tech Final Year Project).

---

**Built with ❤️ using PyTorch, Flask, React, and Tailwind CSS**
