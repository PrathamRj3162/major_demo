#!/bin/bash
# Start both backend and frontend with one command

echo "🚀 Starting Backend..."
cd backend && source venv/bin/activate && python app.py &

echo "🚀 Starting Frontend..."
cd frontend && npm run dev &

wait
