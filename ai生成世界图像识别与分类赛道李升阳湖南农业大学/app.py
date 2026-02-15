import os
import json
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import torch
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import base64
import io
import threading

from src.inference import run, base_transform, tta_variants
from src.model import create_model
from src.utils import list_images, infer_label_from_path
from src.cam import GradCAM, overlay_cam, cam_to_mask, mask_to_bbox
from src.yolov8_wrapper import predict_yolov8_seg
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detection_model = None
segmentation_model = None
transform = None

_models_loaded = False
_models_lock = threading.Lock()

def load_models():
    global detection_model, segmentation_model, transform
    
    # Load detection model (EfficientNet preferred)
    det_weights_path = Path('models/det_best.pt')
    if det_weights_path.exists():
        loaded = False
        try:
            candidate = create_model(model_name='efficientnetv2_s', pretrained=False)
            try:
                state = torch.load(det_weights_path, map_location=device, weights_only=True)
            except (TypeError, RuntimeError):
                state = torch.load(det_weights_path, map_location=device)

            if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
                state = state['state_dict']
            if isinstance(state, dict) and 'model' in state and hasattr(state['model'], 'state_dict'):
                state = state['model'].state_dict()

            try:
                candidate.load_state_dict(state)
            except Exception:
                candidate.load_state_dict(state, strict=False)

            detection_model = candidate.to(device).eval()
            loaded = True
            print("EfficientNet detection model loaded successfully!")
        except Exception as e:
            print(f"Warning: EfficientNet detection model load failed: {e}")

        if not loaded and YOLO is not None:
            try:
                detection_model = YOLO(str(det_weights_path))
                loaded = True
                print("YOLO detection model loaded successfully!")
            except Exception as e:
                print(f"Warning: YOLO detection model load failed: {e}")
                detection_model = None
    
    # Load segmentation model (YOLOv8)
    seg_weights_path = Path('models/seg_best.pt')
    if seg_weights_path.exists():
        try:
            segmentation_model = predict_yolov8_seg(seg_weights_path)
        except Exception as e:
            print(f"Warning: Segmentation model load failed: {e}")
            segmentation_model = None
    
    transform = base_transform(img_size=224)
    print("Models initialization completed!")


def ensure_models_loaded():
    global _models_loaded
    if _models_loaded:
        return
    with _models_lock:
        if _models_loaded:
            return
        load_models()
        _models_loaded = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def detect_tumor():
    try:
        ensure_models_loaded()

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = Path(app.config['UPLOAD_FOLDER']) / unique_filename
        file.save(filepath)
        
        # Process image
        img = Image.open(filepath).convert('RGB')
        
        # Detection inference
        if detection_model is None:
            return jsonify({'error': 'Detection model not loaded'}), 500

        if hasattr(detection_model, 'predict'):  # YOLO model
            # Use YOLO for detection
            results = detection_model.predict(str(filepath), conf=0.5, verbose=False)
            pred = 1 if results and len(results[0].boxes) > 0 else 0
            confidence = 0.0
            prob = 0.0
            
            if pred == 1:
                # Get the highest confidence detection
                boxes = results[0].boxes
                if len(boxes) > 0:
                    confidence = float(boxes.conf.max().cpu().item())
                    prob = confidence
        else:  # EfficientNet model
            # Use EfficientNet for detection
            variants = tta_variants(img)
            batch = torch.stack([transform(v) for v in variants]).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = detection_model(batch).squeeze(1)
                    probs = torch.sigmoid(logits)
            
            prob = float(probs.mean().cpu().item())
            pred = 1 if prob >= 0.5 else 0
            confidence = prob if pred == 1 else 1 - prob
        
        result = {
            'filename': filename,
            'prediction': 'tumor_detected' if pred == 1 else 'no_tumor',
            'confidence': confidence,
            'probability': prob
        }
        
        # Generate visualization if tumor detected
        if pred == 1:
            if hasattr(detection_model, 'predict'):  # YOLO model
                # Use YOLO detections for visualization
                results = detection_model.predict(str(filepath), conf=0.5, verbose=False)
                if results and len(results[0].boxes) > 0:
                    boxes = results[0].boxes
                    img_with_bbox = img.copy()
                    img_array = np.array(img_with_bbox)
                    
                    # Draw all detected boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        cv2.rectangle(img_array, (x1, y1), (x2, y2), (255, 0, 0), 3)
                        cv2.putText(img_array, f'{conf:.2f}', (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    result['bbox_image'] = image_to_base64(Image.fromarray(img_array))
                    result['bbox'] = [int(x1), int(y1), int(x2), int(y2)]
            else:  # EfficientNet model
                # Generate CAM visualization
                x = transform(img).unsqueeze(0).to(device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    cam = GradCAM(detection_model).generate(x)
                
                # Generate CAM visualization
                cam_vis = overlay_cam(img, cam)
                result['cam_image'] = image_to_base64(cam_vis)
                
                # Generate mask
                mask_small = cam_to_mask(cam, percentile=80.0)
                w, h = img.size
                mask_img = Image.fromarray((mask_small * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)
                result['mask_image'] = image_to_base64(mask_img)
                
                # Generate bounding box
                mask_arr = np.array(mask_img)
                bbox = mask_to_bbox(mask_arr)
                if bbox:
                    result['bbox'] = bbox
                    
                    # Draw bounding box on original image
                    img_with_bbox = img.copy()
                    img_arr = np.array(img_with_bbox)
                    cv2.rectangle(
                        img_arr,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (255, 0, 0),
                        3,
                    )
                    result['bbox_image'] = image_to_base64(Image.fromarray(img_arr))
        
        # Save original image as base64
        result['original_image'] = image_to_base64(img)
        
        # Segmentation if available
        if segmentation_model and pred == 1:
            try:
                seg_result = segmentation_model.predict(str(filepath), save=False)
                if seg_result and len(seg_result) > 0:
                    # Process segmentation result
                    seg_img = Image.open(filepath).convert('RGB')
                    result['segmentation_available'] = True
            except Exception as seg_error:
                print(f"Segmentation error: {seg_error}")
                result['segmentation_available'] = False
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_detect', methods=['POST'])
def batch_detect():
    try:
        ensure_models_loaded()

        data = request.get_json()
        if not data or 'image_paths' not in data:
            return jsonify({'error': 'No image paths provided'}), 400
        
        image_paths = data['image_paths']
        results = []
        
        for img_path in image_paths:
            try:
                full_path = Path(img_path)
                if not full_path.exists() or not allowed_file(full_path.name):
                    continue
                
                img = Image.open(full_path).convert('RGB')
                
                # Detection inference
                if detection_model is None:
                    raise RuntimeError('Detection model not loaded')

                if hasattr(detection_model, 'predict'):
                    yolo_results = detection_model.predict(str(full_path), conf=0.5, verbose=False)
                    pred = 1 if yolo_results and len(yolo_results[0].boxes) > 0 else 0
                    if pred == 1 and yolo_results and len(yolo_results[0].boxes) > 0:
                        conf = float(yolo_results[0].boxes.conf.max().cpu().item())
                        prob = conf
                    else:
                        prob = 0.0
                else:
                    variants = tta_variants(img)
                    batch = torch.stack([transform(v) for v in variants]).to(device)

                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                            logits = detection_model(batch).squeeze(1)
                            probs = torch.sigmoid(logits)

                    prob = float(probs.mean().cpu().item())
                    pred = 1 if prob >= 0.5 else 0
                
                results.append({
                    'filename': full_path.name,
                    'path': str(full_path),
                    'prediction': 'tumor_detected' if pred == 1 else 'no_tumor',
                    'probability': prob,
                    'confidence': prob if pred == 1 else 1 - prob
                })
                
            except Exception as e:
                results.append({
                    'filename': Path(img_path).name,
                    'path': img_path,
                    'error': str(e)
                })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    try:
        ensure_models_loaded()

        detection_type = 'Unknown'
        if detection_model is None:
            detection_type = 'NotLoaded'
        elif hasattr(detection_model, 'predict'):
            detection_type = 'YOLO'
        else:
            detection_type = 'EfficientNetV2-S'

        info = {
            'device': str(device),
            'detection_model_loaded': detection_model is not None,
            'segmentation_model_loaded': segmentation_model is not None,
            'model_types': {
                'detection': detection_type,
                'segmentation': 'YOLOv8'
            }
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def results_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Loading models...")
    load_models()
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
