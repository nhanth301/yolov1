from model import Yolov1
from loss import YoloLoss
from dataset import CustomVOCDataset
from torch import optim
from torch.utils.data import DataLoader
import torch
from config import LEARNING_RATE, DEVICE, LOAD_MODEL, LOAD_MODEL_FILE, CLASS_MAPPING, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, EPOCHS
from utils import load_checkpoint, save_checkpoint
from train_utils import get_train_transforms, get_valid_transforms, train_fn, test_fn
from termcolor import colored

def train():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = CustomVOCDataset(root='./data',
                                     year='2012',
                                     image_set='trainval',
                                     download=True,
                                     )

    test_dataset = CustomVOCDataset(root='./data',
                                    year='2012',
                                    image_set='val',
                                    download=True,
                                    )

    train_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_train_transforms())
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    best_mAP_train = 0
    best_mAP_test = 0

    for epoch in range(EPOCHS):
        train_mAP = train_fn(train_loader, model, optimizer, loss_fn, epoch)
        test_mAP = test_fn(test_loader, model, loss_fn, epoch)

        if train_mAP > best_mAP_train:
            best_mAP_train = train_mAP

        if test_mAP > best_mAP_test:
            best_mAP_test = test_mAP

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)


    print(colored(f"Best Train mAP: {best_mAP_train:3.10f}", 'green'))
    print(colored(f"Best Test mAP: {best_mAP_test:3.10f}", 'yellow'))


if __name__ == '__main__':
    train()