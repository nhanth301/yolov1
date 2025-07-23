import numpy as np

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
