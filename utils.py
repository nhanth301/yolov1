import numpy as np
import torch 
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


def intersection_over_union(boxes_preds, boxes_labels):
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2   
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
    