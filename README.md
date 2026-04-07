# AI-Driven Precision Diagnostics System

## Pneumonia Detection using Deep Learning, Explainable AI & Federated Learning

B.Tech Major Project — Delhi Technological University

**Frontend (Vercel)**: https://pneumoai-diagnostics.vercel.app/  
**Backend API (Render)**: https://major-demo-1.onrender.com/api/health  

> Note: The Render free tier goes to sleep after 15 min idle. First request might take ~30 seconds to wake up.

---

## What this project does

We built a web-based system that takes chest X-ray images as input and predicts whether the patient has pneumonia or not. It uses:

- **DenseNet121** (pretrained on ImageNet, transfer learning) for classification
- **Grad-CAM** to show which regions of the X-ray the model is focusing on — this is the explainability part
- **Federated Learning (FedAvg)** simulation to show how the model could be trained across multiple hospitals without sharing patient data

The frontend is built with React + Vite and the backend runs on Flask with PyTorch.

---

## Project Structure

```
btech_major_final_project/
├── backend/
│   ├── app.py                  # main flask app
│   ├── config.py               # all constants and paths
│   ├── requirements.txt
│   ├── models/
│   │   ├── densenet_model.py   # DenseNet121 architecture
│   │   └── model_loader.py     # loads and caches the model
│   ├── services/
│   │   ├── prediction.py       # runs inference
│   │   ├── gradcam.py          # generates heatmaps
│   │   └── federated.py        # FedAvg simulation
│   ├── routes/
│   │   ├── predict_routes.py   # /api/predict, /api/upload
│   │   ├── gradcam_routes.py   # /api/gradcam
│   │   ├── federated_routes.py # /api/federated-train
│   │   └── stats_routes.py     # /api/model-stats
│   ├── utils/
│   │   ├── preprocessing.py    # image resizing & normalization
│   │   └── helpers.py          # base64, filenames, etc.
│   └── uploads/                # temp folder for uploaded images
├── frontend/
│   ├── src/
│   │   ├── components/         # reusable UI components
│   │   ├── pages/              # dashboard, upload, results, etc.
│   │   ├── services/api.js     # axios calls to backend
│   │   ├── App.jsx
│   │   └── index.css
│   ├── vite.config.js
│   └── tailwind.config.js
├── data/                       # dataset info (see data/README.md)
└── README.md
```

---

## How to run locally

### Requirements
- Python 3.9+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Backend starts on http://localhost:5001

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend starts on http://localhost:5173

---

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|-------------|
| GET | /api/health | check if server is up |
| POST | /api/upload | upload an xray image |
| POST | /api/predict | upload + predict + gradcam all in one |
| POST | /api/gradcam | just the gradcam heatmap |
| POST | /api/federated-train | run FedAvg simulation |
| GET | /api/model-stats | model performance numbers |

---

## How the AI pipeline works

1. User uploads a chest X-ray through the UI
2. Image is resized to 224x224, converted to RGB, normalized using ImageNet mean/std
3. DenseNet121 runs a forward pass, softmax gives class probabilities
4. If confidence < 70%, it flags "Needs Review" (for clinical safety)
5. Grad-CAM hooks into the last DenseBlock, captures activations + gradients, generates a heatmap overlay
6. Everything is sent back to the frontend — prediction, confidence, probabilities, and the Grad-CAM images

## Federated Learning

We simulate FedAvg across multiple hospital clients:
- Server sends global model to all clients
- Each client trains on its local data (we use synthetic data for the demo)
- Clients send back model weights (not the data itself — thats the privacy part)
- Server does a weighted average (weighted by dataset size)
- This repeats for N rounds

We use a smaller CNN instead of DenseNet for this because training DenseNet on CPU would take forever. The smaller model still shows convergence properly.

---

## Pages in the UI

1. **Dashboard** — overview, model info, quick links
2. **Upload** — drag and drop an xray, click analyze
3. **Results** — shows prediction, confidence gauge, Grad-CAM overlay
4. **Federated Learning** — configurable simulation with charts
5. **Performance** — confusion matrix, loss/acc curves, ROC curve

---

## Dataset

We used the Chest X-Ray Pneumonia dataset from Kaggle (Kermany et al.):
- https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- ~5800 images, 2 classes (NORMAL / PNEUMONIA)
- The system works without fine-tuned weights too (just uses ImageNet defaults), but accuracy is much better after training on this dataset

---

## Tech Stack

- **Backend**: Flask, PyTorch, torchvision, OpenCV, Pillow
- **Frontend**: React 18, Vite, Tailwind CSS, Recharts, Framer Motion, Lucide icons, Axios
- **Deployment**: Render (backend), Vercel (frontend)

---

## Deployment

### Backend on Render
- Create a Web Service, set root directory to `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn "app:create_app()" --bind 0.0.0.0:$PORT --timeout 300`

### Frontend on Vercel
- Import the repo, set framework to Vite, root directory to `frontend`
- Add env variable: `VITE_API_URL` = your Render backend URL + `/api`

---

This project was built as part of our B.Tech final year major project at DTU.
