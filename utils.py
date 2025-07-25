import numpy as np
import torch 
import torch.nn as nn
from collections import Counter
from matplotlib import patches
import matplotlib.pyplot as plt
from small_model import Yolov1Small, CNNBlock
from qat_model import Yolov1SmallQAT
import cv2
import os
from natsort import natsorted

def convert_to_yolo_format(target, width, height, class_mapping):
    annotations = target['annotation']['object']

    r_width = int(target['annotation']['size']['width'])
    r_height = int(target['annotation']['size']['height'])

    if not isinstance(annotations, list):
        annotations = [annotations]
    
    boxes = []

    for anno in annotations:
        xmin = int(anno['bndbox']['xmin']) / r_width
        xmax = int(anno['bndbox']['xmax']) / r_width
        ymin = int(anno['bndbox']['ymin']) / r_height
        ymax = int(anno['bndbox']['ymax']) / r_height

        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin

        class_name = anno['name']
        class_id = class_mapping[class_name] if class_name in class_mapping else 0

        boxes.append([class_id, x_center, y_center, width, height])

    return np.array(boxes)


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2   
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == 'corners': 
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]
    
    else:
        return None

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    [class_pred, prob_score, x1, y1, x2, y2]
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=20):
    """
        [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    """
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_iou_idx = idx
                
            if best_iou > iou_threshold:
                    if amount_bboxes[detection[0]][best_iou_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_iou_idx] = 1
                    else:
                        FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "More values than x, y, w, h in a box"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.show()

def convert_cellboxes(predictions, S=7):

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def get_bboxes_training(
    outputs,
    labels,
    iou_threshold=0.5,
    threshold=0.4,
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    train_idx = 0

    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(outputs)

    for idx in range(outputs.shape[0]):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[idx]:
            if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)

        train_idx += 1

    return all_pred_boxes, all_true_boxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def plot_image_with_labels(image, ground_truth_boxes, predicted_boxes, class_mapping):

    inverted_class_mapping = {v: k for k, v in class_mapping.items()}

    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in ground_truth_boxes:
        label_index, box = box[0], box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)
        class_name = inverted_class_mapping.get(label_index, "Unknown")
        ax.text(upper_left_x * width, upper_left_y * height, class_name, color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.2))

    for box in predicted_boxes:
        label_index, box = box[0], box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
        class_name = inverted_class_mapping.get(label_index, "Unknown")
        ax.text(upper_left_x * width, upper_left_y * height, class_name, color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.2))

    plt.show()



def fuse_original_model_for_qat(model):
    if hasattr(model, 'darknet') and isinstance(model.darknet, nn.Sequential):
        for i, sub_module in enumerate(model.darknet):
            if isinstance(sub_module, CNNBlock):
                torch.quantization.fuse_modules(sub_module, ['conv', 'batchnorm'], inplace=True)
    return model

def load_quantized_model_for_inference(
    path_to_quantized_state_dict: str,
    model_kwargs: dict,
    qconfig_backend: str = 'fbgemm' 
):
    original_yolov1_small = Yolov1Small(**model_kwargs)
    original_yolov1_small.eval()
    
    fused_original_model = fuse_original_model_for_qat(original_yolov1_small)
    qat_model = Yolov1SmallQAT(fused_original_model)
    qat_model.qconfig = torch.quantization.get_default_qconfig(qconfig_backend)
    torch.quantization.prepare_qat(qat_model, inplace=True)
    

    print("Attempting to load the fully quantized model...")

    qat_model.eval() 
    qat_model_cpu = qat_model.to('cpu')
    converted_model_structure = torch.quantization.convert(qat_model_cpu, inplace=False)
    converted_model_structure.load_state_dict(
        torch.load(path_to_quantized_state_dict, map_location='cpu')
    )
    print(f"Successfully loaded fully quantized model from {path_to_quantized_state_dict}")
    
    converted_model_structure.eval()
    return converted_model_structure

def export_qat2onnx(qat_model, onnx_path, input_shape=(1, 3, 448, 448)):
    dummy_input_export = torch.randn(input_shape).to('cpu')
    torch.onnx.export(qat_model,
                      dummy_input_export,
                      onnx_path,
                      opset_version=13, 
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}
                     )
    print(f"Quantized model exported to {onnx_path}")


def create_video_from_images(image_folder, output_path='video/output_video.mp4', fps=30):
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = natsorted(images)

    if not images:
        return

    first_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_path)
    if frame is None:
        return

    height, width, _ = frame.shape

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    count = 0
    for img in images:
        img_path = os.path.join(image_folder, img)
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        frame = cv2.resize(frame, (width, height))
        out.write(frame)
        count += 1

    out.release()

def plot_single_frame_with_boxes(frame, boxes, class_mapping):
    CLASS_NAMES = [None] * len(class_mapping)
    for name, idx in class_mapping.items():
        CLASS_NAMES[idx] = name
    h, w, _ = frame.shape
    for box in boxes:
        class_pred = int(box[0])
        prob = box[1]
        x, y, bw, bh = box[2], box[3], box[4], box[5]
        
        x1 = int((x - bw / 2) * w)
        y1 = int((y - bh / 2) * h)
        x2 = int((x + bw / 2) * w)
        y2 = int((y + bh / 2) * h)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"{CLASS_NAMES[class_pred]}: {prob:.2f}"
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame

if __name__ == '__main__':
    create_video_from_images('data2007/VOCdevkit/VOC2007/JPEGImages')