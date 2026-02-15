from ultralytics import YOLO

def train():
    model = YOLO('yolov8n-seg.pt')
    model.train(data='yolo_seg.yaml', epochs=50, imgsz=640, project='runs_seg_train', name='yolov8n_seg_auto')

if __name__ == '__main__':
    train()
