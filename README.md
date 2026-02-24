# ArcFace Facial Recognition System with Attendance Tracking

Repository: https://github.com/gagan27scoriant/face_recognition_arcface.git

A comprehensive facial recognition system built with **ArcFace** and **DeepFace**, featuring real-time face detection, enrollment, recognition, and automated attendance logging with a modern web-based interface.

## 🎯 Features

- **Real-time Face Recognition**: Live camera feed with instant face detection and identification
- **ArcFace Model**: State-of-the-art deep learning model (ResNet-50, 512-dimensional embeddings)
- **Automated Attendance Logging**: One-time-per-day attendance tracking with dated CSV files
- **Smart Unknown Face Handling**: Automatically assigns unique IDs to unknown persons and recognizes them on subsequent appearances
- **Model Training**: Easy-to-use training pipeline for enrolling new faces
- **Web Dashboard**: Modern responsive Flask web interface with real-time updates
- **CSV-based Storage**: Daily attendance records with automatic file rotation
- **GPU/CPU Auto-detection**: Seamlessly switches between GPU and CPU based on availability

## 📋 System Requirements

- **OS**: Linux/macOS/Windows
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: 2GB (includes pre-trained models)
- **Webcam**: Any standard USB or built-in camera
- **GPU** (Optional): CUDA-capable GPU for faster processing

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd face_recognition_arcface
```

### 2. Create Virtual Environment

```bash
python3 -m venv face_venu
source face_venu/bin/activate  # On Windows: face_venu\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The system automatically downloads pre-trained models on first run:
- ArcFace model (ResNet-50)
- RetinaFace detector
- VGGFace2 embeddings

## 🚀 Quick Start

### 1. Train the Model with Known Faces

Organize your face images in the `dataset/` directory:

```
dataset/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
└── personN/
    └── ...
```

Train the model:

```bash
python train.py
```

This will:
- Extract face embeddings from all training images
- Create `models/arcface_model.pkl` with trained embeddings
- Save model metadata and statistics

### 2. Start the Application

```bash
python app.py
```

The app will:
- Initialize the Flask server on `http://localhost:5000`
- Load the pre-trained ArcFace model
- Start the facial recognition engine
- Create today's attendance CSV file: `database/Attendance_YYYY-MM-DD.csv`

### 3. Access the Web Dashboard

Open your browser and navigate to:

```
http://localhost:5000
```

**Dashboard Features:**
- **Camera Preview** (Left side): Live video feed with face detection boxes
- **Registered Persons** (Right top): List of all trained face identities
- **Today's Entries** (Right bottom): Real-time attendance records
- **Controls**: Retrain Model, Download Attendance CSV

## 📁 Project Structure

```
face_recognition_arcface/
├── app.py                      # Main Flask application
├── train.py                    # Model training script
├── config.py                   # Configuration settings
├── requirements.txt            # Dependencies
├── database/
│   ├── Attendance_YYYY-MM-DD.csv  # Daily attendance records
│   └── attendance.db           # SQLite database (legacy)
├── dataset/
│   ├── person1/               # Training images for person 1
│   ├── person2/               # Training images for person 2
│   └── ...
├── models/
│   └── arcface_model.pkl      # Trained embeddings
├── static/
│   └── style.css              # Web interface styling
├── templates/
│   └── index.html             # Web dashboard template
├── utils/
│   ├── embedding_utils.py     # Face embedding extraction
│   ├── logger.py              # Attendance logging logic
│   ├── recognition.py         # Face recognition engine
│   └── logger.py              # Logging utilities
└── face_venu/                 # Virtual environment (created during setup)
```

## ⚙️ Configuration

Edit `config.py` to customize system behavior:

```python
THRESHOLD = 0.65                    # Face matching confidence threshold (0-1)
DETECTOR_BACKEND = "opencv"         # Face detector: opencv, retinaface, mtcnn
MIN_FACE_SIZE = 20                  # Minimum face size in pixels
GPU_ENABLED = True                  # Enable GPU acceleration (auto-detected)
```

**Key Parameters:**
- **THRESHOLD**: Lower = more lenient matching, Higher = stricter matching
- **Adjust for your use case**: 
  - Attendance: 0.60-0.65 (recommended)
  - Security: 0.70+ (strict)
  - General use: 0.65

## 🎥 How It Works

### Face Recognition Pipeline

```
Camera Frame
    ↓
Face Detection (RetinaFace)
    ↓
Face Alignment & Normalization
    ↓
Embedding Extraction (ArcFace)
    ↓
Similarity Matching against Trained Faces
    ↓
Decision: Recognized Person or Unknown?
    ↓
Attendance Logging (One-time per day)
    ↓
CSV Database Update
```

### Attendance Logic (One-Time-Per-Day)

1. **Person Recognized**: Check if already logged today
   - **Already Logged** → Skip (no duplicate entry)
   - **Not Logged Yet** → Log to CSV with timestamp

2. **Unknown Person**: Automatically assign unique ID
   - First unknown → `Unique_01`
   - Same unknown appears again → Same `Unique_01` (smart matching)
   - New unknown → `Unique_02`

3. **Daily Reset**: New date = new CSV file
   - Same person can be logged again tomorrow
   - Automatic archival of previous day's records

## 📊 CSV File Format

Daily attendance files are named: `Attendance_2026-02-24.csv`

**Columns:**
| Name | Date | Time | Confidence | Status |
|------|------|------|-----------|--------|
| Sai_Kishen_Vukku | 2026-02-24 | 14:30:45 | 0.92 | Recognized |
| Unique_01 | 2026-02-24 | 14:32:12 | 0.76 | Unknown |
| Person2 | 2026-02-24 | 14:35:20 | 0.88 | Recognized |

## 🎓 API Endpoints

### Get Attendance Records

```bash
GET /api/attendance
```

**Response:**
```json
{
  "date": "2026-02-24",
  "total_entries": 3,
  "unique_persons": 3,
  "avg_confidence": 0.85,
  "records": [
    {
      "name": "Sai_Kishen_Vukku",
      "time": "14:30:45",
      "confidence": 0.92,
      "status": "Recognized"
    }
  ]
}
```

### Get System Status

```bash
GET /api/status
```

**Response:**
```json
{
  "status": "running",
  "model_loaded": true,
  "camera_active": true,
  "threshold": 0.65,
  "gpu_available": false
}
```

### Get Registered Persons

```bash
GET /api/persons
```

**Response:**
```json
{
  "count": 2,
  "persons": ["Sai_Kishen_Vukku", "Person2"]
}
```

## 🔧 Training Your Own Model

### Step 1: Prepare Training Data

Create folders with person names and add their face images:

```
dataset/
├── Sai_Kishen_Vukku/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── photo3.jpg
│   └── ...  (5-10 images recommended)
├── John_Doe/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
```

**Image Requirements:**
- Format: JPG, PNG, or BMP
- Clear, frontal face view
- Good lighting conditions
- Minimum 20x20 pixels
- 5-10 images per person recommended

### Step 2: Run Training

```bash
python train.py
```

**Output:**
```
Loading dataset from: dataset/
Processing: Sai_Kishen_Vukku (8 images)
Processing: John_Doe (7 images)
...
Training complete!
Model saved to: models/arcface_model.pkl
Embeddings extracted: 15 faces
```

### Step 3: Test Recognition

Start the app and test face recognition in real-time:

```bash
python app.py
```

## 📈 Usage Scenarios

### Scenario 1: Classroom Attendance

1. Train model with student photos
2. Start app at beginning of class
3. Each student appears on camera → auto-logged once
4. Download CSV at end of class
5. Integrate with attendance system

### Scenario 2: Office Entry/Exit

1. Train model with employee photos
2. Run app at office entrance
3. Employees identified → logged with timestamp
4. Generate daily attendance reports
5. Track attendance across multiple days

### Scenario 3: Event Check-in

1. Pre-train model with registered guests
2. Run at event entrance
3. Unknown guests get automatic Unique_IDs
4. Track attendance and unknown visitors
5. Generate check-in report

## 🐛 Troubleshooting

### Issue: No faces detected in camera

**Solution:**
- Ensure good lighting conditions
- Face should be clearly visible and frontal
- Increase camera resolution
- Check webcam permissions in OS settings

### Issue: High false positive rate

**Solution:**
- Increase `THRESHOLD` in `config.py` (0.65 → 0.70)
- Add more diverse training images
- Ensure clear face images without masks/glasses

### Issue: High false negative rate

**Solution:**
- Decrease `THRESHOLD` in `config.py` (0.65 → 0.60)
- Retrain model with better images
- Ensure consistent lighting between training and test

### Issue: Slow performance

**Solution:**
- Check GPU availability: `nvidia-smi`
- Set `GPU_ENABLED = True` in `config.py`
- Reduce video frame resolution
- Close other memory-intensive applications

### Issue: Camera not starting

**Solution:**
```bash
# Check camera permissions
ls -la /dev/video0

# List available cameras
v4l2-ctl --list-devices

# Test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

## 📝 Logging and Debug

View application logs:

```bash
tail -f /tmp/app.log             # Real-time logs
cat /tmp/app.log | grep ERROR    # Show errors only
```

Enable debug mode in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

## 🔐 Security Notes

- **No external data storage**: All data stored locally
- **No cloud upload**: Attendance stays on your machine
- **Local processing**: Face embeddings processed on-device
- **HTTPS ready**: Can be deployed with SSL certificates

## 📊 Performance Metrics

Typical performance (CPU):
- Face detection: ~100ms per frame
- Embedding extraction: ~150ms per face
- Recognition: ~50ms per comparison
- FPS: 5-10 (depends on face count and CPU)

With GPU:
- Face detection: ~20ms per frame
- Embedding extraction: ~30ms per face
- Recognition: ~10ms per comparison
- FPS: 20-30+

## 🤝 Contributing

To improve the system:

1. Collect more training data for existing persons
2. Test with different lighting conditions
3. Add new features to the Flask dashboard
4. Optimize performance for your hardware

## 📜 License

This project uses:
- **DeepFace**: MIT License
- **TensorFlow**: Apache 2.0 License
- **OpenCV**: BSD License

## ❓ FAQ

**Q: Can I run this on GPU?**
A: Yes! Install CUDA toolkit and TensorFlow-GPU. The system auto-detects GPU availability.

**Q: How many faces can the system handle?**
A: Tested with 100+ trained identities. Performance depends on hardware.

**Q: Can I export attendance data?**
A: Yes! CSV files can be downloaded directly from the dashboard or accessed from `database/Attendance_YYYY-MM-DD.csv`.

**Q: Is real-time performance possible?**
A: Yes, with GPU. Achieves 20-30 FPS. On CPU, expect 5-10 FPS.

**Q: Can I use this for security?**
A: For basic identification yes, but increase threshold (0.70+) for security-critical applications.

**Q: Do I need to retrain if I add new people?**
A: Yes. Run `python train.py` after adding new face images to the dataset.

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review application logs
3. Verify camera and lighting conditions
4. Ensure all dependencies are installed correctly

## 🎉 Getting Started

```bash
# 1. Setup
python3 -m venv face_venu
source face_venu/bin/activate
pip install -r requirements.txt

# 2. Add training data
# Place face images in dataset/person_name/ folders

# 3. Train
python train.py

# 4. Run
python app.py

# 5. Access dashboard
# Open http://localhost:5000 in browser
```

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Production Ready ✅
