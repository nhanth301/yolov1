import torchvision
from utils import convert_to_yolo_format
import numpy as np
import torch 

class CustomVOCDataset(torchvision.datasets.VOCDetection):
    def init_config_yolo(self, class_mapping, S=7, B=2, C=20, custom_transforms=None):
        self.S = S  
        self.B = B 
        self.C = C 
        self.class_mapping = class_mapping  
        self.custom_transforms = custom_transforms

    def __getitem__(self, index):
        image, target = super(CustomVOCDataset, self).__getitem__(index)
        img_width, img_height = image.size

        boxes = convert_to_yolo_format(target, img_width, img_height, self.class_mapping)

        just_boxes = boxes[:,1:]
        labels = boxes[:,0]

        if self.custom_transforms:
            sample = {
            'image': np.array(image),
            'bboxes': just_boxes,
            'labels': labels
            }

            sample = self.custom_transforms(**sample)
            image = sample['image']
            boxes = sample['bboxes']
            labels = sample['labels']

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        image  = torch.as_tensor(image, dtype=torch.float32)

        for box, class_label in zip(boxes, labels):
            x, y, width, height = box.tolist()
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix