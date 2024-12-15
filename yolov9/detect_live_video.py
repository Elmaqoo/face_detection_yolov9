import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

def preprocess(frame, input_size=640):
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float()
    return frame_tensor

def run(weights='../best.pt', source=0, imgsz=640, conf_thres=0.25, iou_thres=0.45, device=''):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.stride)

    cap = cv2.VideoCapture(source)  # Use 0 for webcam, or replace with video stream URL

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        input_frame = preprocess(frame, input_size=imgsz).to(device)
        
        with torch.no_grad():
            predictions = model(input_frame)
            predictions = non_max_suppression(predictions, conf_thres, iou_thres, max_det=1000)

        annotator = Annotator(frame, line_width=3, example=str(model.names))
        for det in predictions:
            if len(det):
                det[:, :4] = scale_boxes(input_frame.shape[2:], det[:, :4], frame.shape[:2])
                for *xyxy, conf, cls in reversed(det):
                    label = f'{model.names[int(cls)]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(int(cls)))

        annotated_frame = annotator.result()
        cv2.imshow('Live Video', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best.pt', help='model path')
    parser.add_argument('--source', type=int, default=0, help='camera source, default is 0 for webcam')
    parser.add_argument('--imgsz', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e., 0 or cpu')
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
