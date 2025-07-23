from model import Yolov1
import torch
from config import DEVICE, LOAD_MODEL, LOAD_MODEL_FILE, CLASS_MAPPING, NUM_WORKERS, BATCH_SIZE, PIN_MEMORY
from dataset import CustomVOCDataset
from torch.utils.data import DataLoader
from train_utils import get_valid_transforms
from utils import cellboxes_to_boxes, non_max_suppression, plot_image_with_labels

def test():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_MODEL_FILE)['state_dict'])

    test_dataset = CustomVOCDataset(root='./data', image_set='val', download=False)
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False, drop_last=False)

    model.eval()
    for x, y in test_loader:
        x = x.to(DEVICE)
        out = model(x)

        pred_bboxes = cellboxes_to_boxes(out)
        gt_bboxes = cellboxes_to_boxes(y)

        for idx in range(8):
            pred_box = non_max_suppression(pred_bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
            gt_box = non_max_suppression(gt_bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

            image = x[idx].permute(1,2,0).to("cpu")/255
            plot_image_with_labels(image, gt_box, pred_box, CLASS_MAPPING)

        break 