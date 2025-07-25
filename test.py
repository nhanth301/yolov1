from model import Yolov1
import torch
from config import DEVICE, LOAD_MODEL_FILE, CLASS_MAPPING, NUM_WORKERS, BATCH_SIZE, PIN_MEMORY
from dataset import CustomVOCDataset
from torch.utils.data import DataLoader, Subset
from train_utils import get_valid_transforms, test_fn, test_fn_onnx
from utils import (
    cellboxes_to_boxes, non_max_suppression, plot_image_with_labels,
    load_quantized_model_for_inference, plot_single_frame_with_boxes
)
from loss import YoloLoss
from torchsummary import summary
import cv2
import numpy as np
import onnxruntime
from PIL import Image


def vi_test():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    model.load_state_dict(torch.load(LOAD_MODEL_FILE)['state_dict'])

    test_dataset = CustomVOCDataset(root='./data2007', year='2007', image_set='test', download=False)
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        shuffle=False, drop_last=False
    )

    model.eval()
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)
        pred_bboxes = cellboxes_to_boxes(out)
        gt_bboxes = cellboxes_to_boxes(y)

        for idx in range(8):
            pred_box = non_max_suppression(pred_bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            gt_box = non_max_suppression(gt_bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            image = x[idx].permute(1, 2, 0).to("cpu") / 255
            plot_image_with_labels(image, gt_box, pred_box, CLASS_MAPPING)
        break


def test():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    loss_fn = YoloLoss()
    model.load_state_dict(torch.load(LOAD_MODEL_FILE)['state_dict'])

    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    test_dataset = CustomVOCDataset(root='./data', image_set='val', download=False)
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())
    subset_test_dataset = Subset(test_dataset, list(range(200)))

    test_loader = DataLoader(
        dataset=subset_test_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        shuffle=False, drop_last=False
    )

    quantized_model.eval()
    print(summary(model, (3, 448, 448), device='cpu'))
    print(summary(quantized_model, (3, 448, 448), device='cpu'))


def qat_test():
    print(f"Using device: {DEVICE}")
    model_kwargs = {"split_size": 7, "num_boxes": 2, "num_classes": 20}
    path = 'ckpts/yolov1_small_quantized_qat_final.pth'

    try:
        model = load_quantized_model_for_inference(path, model_kwargs)
        model = model.to('cpu')
        print("Loaded quantized model moved to CPU for inference.")
    except FileNotFoundError:
        print(f"Error: Model state_dict not found at {path}.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    loss_fn = YoloLoss()
    test_dataset = CustomVOCDataset(root='./data', image_set='val', download=False)
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())
    subset_test_dataset = Subset(test_dataset, list(range(200)))

    test_loader = DataLoader(
        dataset=subset_test_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        shuffle=False, drop_last=False
    )

    import time
    start_time = time.time()
    test_fn(test_loader, model, loss_fn)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f}s")


def onnx_test():
    loss_fn = YoloLoss()
    test_dataset = CustomVOCDataset(root='./data', image_set='val', download=False)
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())
    subset_test_dataset = Subset(test_dataset, list(range(200)))

    test_loader = DataLoader(
        dataset=subset_test_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        shuffle=False, drop_last=False
    )

    import time
    start_time = time.time()
    test_fn_onnx(test_loader, 'ckpts/model.onnx', loss_fn)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f}s")


def run_onnx_on_video(video_path, onnx_model_path, class_mapping=CLASS_MAPPING, output_path="video/output_result.mp4"):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import time

    try:
        session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    transform = A.Compose([
        A.Resize(height=448, width=448),
        ToTensorV2()
    ])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = 200

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        original_frame = frame.copy()
        resized = cv2.resize(frame, (448, 448))
        transformed = transform(image=resized)
        input_tensor = transformed["image"].unsqueeze(0)
        input_np = input_tensor.numpy().astype(np.float32)

        outputs = session.run([output_name], {input_name: input_np})
        out = torch.from_numpy(outputs[0])
        bboxes = cellboxes_to_boxes(out)[0]
        bboxes = non_max_suppression(bboxes, iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        frame_with_boxes = plot_single_frame_with_boxes(original_frame, bboxes, class_mapping)
        out_writer.write(frame_with_boxes)

        elapsed = time.time() - start_time
        print(f"Frame {frame_idx + 1} | Time: {elapsed:.3f}s | FPS: {1 / elapsed:.2f}")
        frame_idx += 1

    cap.release()
    out_writer.release()
    print(f"Video saved to: {output_path}")


if __name__ == '__main__':
    run_onnx_on_video('video/output_video.mp4', 'ckpts/model.onnx')
