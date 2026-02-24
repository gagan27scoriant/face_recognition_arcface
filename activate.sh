#!/bin/bash
# Quick activation script for face_venu environment

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Activating face_venu environment..."
source "$PROJECT_DIR/face_venu/bin/activate"

echo "✓ Virtual environment activated!"
echo "✓ Location: $VIRTUAL_ENV"
echo ""
echo "You can now run:"
echo "  python app.py          → Start web server"
echo "  python train.py        → Train face recognition model"
echo "  python initialize_db.py → Initialize database"
echo ""
echo "To deactivate later, run: deactivate"
