from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

model.train(data='datas.yaml',epochs=5,imgsz=640,batch=24,optimizer='Adam')