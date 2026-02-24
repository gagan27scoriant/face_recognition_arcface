"""
Flask Web Application - Real-Time Facial Recognition System
Uses ArcFace embeddings via DeepFace for face recognition
Supports GPU acceleration with CPU fallback
"""

import os
import sys
import logging
import cv2
import time
import json
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
from datetime import datetime
from threading import Thread, Lock
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import utilities
from config import (
    MODEL_NAME, THRESHOLD, FRAME_SKIP, TARGET_FPS,
    INPUT_SIZE, EMBEDDING_DIM, DATABASE_DIR
)
from utils.device_manager import get_device_manager
from utils.embedding_utils import get_embedding_manager
from utils.recognition import get_face_recognizer
from utils.logger import get_attendance_logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
camera = None
frame_lock = Lock()
current_frame = None
recognition_results = []
fps_counter = defaultdict(lambda: {'count': 0, 'start_time': time.time()})
frame_count = 0

# Initialize components
device_manager = get_device_manager()
embedding_manager = get_embedding_manager()
face_recognizer = get_face_recognizer()
attendance_logger = get_attendance_logger()

# Print system info on startup
logger.info("="*60)
logger.info("FACIAL RECOGNITION SYSTEM - STARTUP")
logger.info("="*60)
device_manager.print_device_info()
logger.info(f"Model: {MODEL_NAME}")
logger.info(f"Threshold: {THRESHOLD}")
logger.info(f"Registered Persons: {embedding_manager.get_database_size()}")
logger.info(f"Persons: {', '.join(embedding_manager.get_all_persons())}")
logger.info("="*60)


def init_camera(camera_index: int = 0) -> bool:
    """Initialize camera"""
    global camera
    try:
        camera = cv2.VideoCapture(camera_index)
        if camera.isOpened():
            logger.info(f"✓ Camera initialized (index: {camera_index})")
            return True
        else:
            logger.error("Failed to initialize camera")
            return False
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        return False


def generate_frames():
    """
    Generate video frames for streaming
    Processes frames with face detection and recognition
    """
    global current_frame, frame_count, recognition_results, camera
    
    frame_skip_count = 0
    detection_interval = FRAME_SKIP
    
    while True:
        try:
            success, frame = camera.read()
            
            if not success:
                logger.warning("Failed to read frame from camera")
                break
            
            frame_count += 1
            frame_skip_count += 1
            
            # Resize frame for processing
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            target_height = 480
            target_width = int(target_height * aspect_ratio)
            frame_resized = cv2.resize(frame, (target_width, target_height))
            
            recognized_faces = []
            
            # Process every Nth frame for performance
            if frame_skip_count >= detection_interval:
                try:
                    recognized_faces = face_recognizer.recognize_faces_in_frame(frame_resized)
                    
                    # Log attendance for recognized faces
                    for face_data in recognized_faces:
                        # Determine person name based on confidence
                        name = face_data['name'] if face_data['confidence'] > THRESHOLD else "Unknown"
                        embedding = face_data.get('embedding')
                        
                        # Logger handles all deduplication logic (ONE TIME PER DAY)
                        success, logged_name = attendance_logger.log_attendance(
                            name,
                            face_data['confidence'],
                            embedding
                        )
                        
                        if success:
                            logger.info(f"✅ Person logged: {logged_name}")
                    
                    recognition_results = recognized_faces
                    frame_skip_count = 0
                    
                except Exception as e:
                    logger.error(f"Recognition error: {e}")
            
            # Draw results on frame
            annotated_frame = face_recognizer.draw_recognition_results(frame_resized, recognized_faces)
            
            # Add FPS counter
            fps = fps_counter['main']
            fps['count'] += 1
            elapsed = time.time() - fps['start_time']
            if elapsed >= 1:
                fps_display = fps['count'] / elapsed
                fps['count'] = 0
                fps['start_time'] = time.time()
            else:
                fps_display = fps['count'] / max(elapsed, 0.001)
            
            # Add status overlay
            status_text = f"FPS: {fps_display:.1f} | Device: {device_manager.device_name} | Detected: {len(recognition_results)}"
            cv2.putText(annotated_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Store current frame
            with frame_lock:
                current_frame = annotated_frame
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            # Stream to client
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            logger.error(f"Frame generation error: {e}")
            continue


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html',
                          model_name=MODEL_NAME,
                          threshold=THRESHOLD,
                          device=device_manager.device_name)


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    """Get system status"""
    try:
        return jsonify({
            'status': 'ok',
            'device': device_manager.device_name,
            'device_type': device_manager.device,
            'model': MODEL_NAME,
            'threshold': THRESHOLD,
            'registered_persons': embedding_manager.get_database_size(),
            'persons': embedding_manager.get_all_persons(),
            'fps': fps_counter['main']['count'],
            'frame_count': frame_count
        })
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/attendance')
def api_attendance():
    """Get today's attendance records"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        report = attendance_logger.get_summary_report(today)
        return jsonify(report)
    except Exception as e:
        logger.error(f"Attendance API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/attendance/<person_name>')
def api_person_attendance(person_name):
    """Get attendance records for a person"""
    try:
        days = request.args.get('days', 7, type=int)
        df = attendance_logger.get_person_attendance(person_name, days)
        return jsonify(df.to_dict('records'))
    except Exception as e:
        logger.error(f"Person attendance API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/reload')
def api_reload():
    """Reload embeddings database"""
    try:
        embedding_manager.load_embeddings()
        face_recognizer.update_database()
        logger.info("✓ Database reloaded")
        return jsonify({
            'status': 'ok',
            'message': 'Database reloaded',
            'registered_persons': embedding_manager.get_database_size()
        })
    except Exception as e:
        logger.error(f"Reload error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/current_detections')
def api_current_detections():
    """Get current face detections"""
    try:
        return jsonify({
            'detections': recognition_results,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Detections API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/download/attendance')
def download_attendance():
    """Download today's attendance CSV file"""
    try:
        from flask import send_file
        csv_file = attendance_logger.attendance_csv
        
        if not os.path.exists(csv_file):
            return jsonify({'status': 'error', 'message': 'No attendance file found'}), 404
        
        return send_file(
            csv_file,
            as_attachment=True,
            download_name=os.path.basename(csv_file),
            mimetype='text/csv'
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/history')
def api_history():
    """Return list of attendance CSV files in the database directory"""
    try:
        files = []
        for fname in os.listdir(DATABASE_DIR):
            if fname.lower().endswith('.csv') and fname.startswith('Attendance_'):
                files.append(fname)
        files.sort(reverse=True)
        return jsonify({'status': 'ok', 'files': files})
    except Exception as e:
        logger.error(f"History API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/download/history/<path:filename>')
def download_history_file(filename: str):
    """Download a specific attendance CSV from history"""
    try:
        from flask import send_file
        safe_name = os.path.basename(filename)
        csv_path = os.path.join(DATABASE_DIR, safe_name)
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404
        return send_file(csv_path, as_attachment=True, download_name=safe_name, mimetype='text/csv')
    except Exception as e:
        logger.error(f"History download error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    logger.error(f"Server error: {error}")
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == "__main__":
    try:
        # Initialize camera
        if not init_camera():
            logger.error("Failed to initialize camera. Exiting.")
            sys.exit(1)
        
        # Start Flask app
        logger.info("Starting Flask application...")
        logger.info("Open http://localhost:5000 in your browser")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        logger.info("Application closed")