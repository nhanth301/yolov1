from model import Yolov1
from loss import YoloLoss
from dataset import CustomVOCDataset
from torch import optim
from torch.utils.data import DataLoader
import torch
from config import LEARNING_RATE, DEVICE, LOAD_MODEL, LOAD_MODEL_FILE, CLASS_MAPPING, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, EPOCHS
from utils import load_checkpoint, save_checkpoint
from train_utils import get_train_transforms, get_valid_transforms, train_fn, test_fn, train_fn_with_mse_distillation, train_fn_qat
from termcolor import colored
from small_model import Yolov1Small
from qat_model import Yolov1SmallQAT, fuse_original_model_for_qat

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

def kd_train():
    model = Yolov1Small(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    teacher_model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    teacher_model.load_state_dict(torch.load('/kaggle/input/yolov1-ckpt/final_yolov1.pth.tar')['state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = YoloLoss()

    train_dataset = CustomVOCDataset(root='./data',
                                     year='2012',
                                     image_set='trainval',
                                     download=False,
                                     )

    test_dataset = CustomVOCDataset(root='./data',
                                    year='2012',
                                    image_set='val',
                                    download=False,
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
        train_mAP = train_fn_with_mse_distillation(train_loader, model, teacher_model, optimizer, loss_fn, epoch)
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

def qat_train():
    model_kwargs = {"split_size": 7, "num_boxes": 2, "num_classes": 20}
    PATH_TO_DISTILLED_MODEL = "/kaggle/input/kdyolov1/KD_yolov1.pth.tar"

    original_yolov1_small = Yolov1Small(**model_kwargs)

    try:
        original_yolov1_small.load_state_dict(torch.load(PATH_TO_DISTILLED_MODEL)['state_dict'])
        print(f"Loaded state_dict from {PATH_TO_DISTILLED_MODEL}")
    except FileNotFoundError:
        print(f"Distilled model not found at {PATH_TO_DISTILLED_MODEL}")
    except Exception as e:
        print(f"Error loading state_dict: {e}")

    original_yolov1_small.eval()
    fused_original_model = fuse_original_model_for_qat(original_yolov1_small)

    qat_model = Yolov1SmallQAT(fused_original_model)
    qat_model = qat_model.to(DEVICE)

    qconfig = torch.quantization.get_default_qconfig('fbgemm')
    qat_model.qconfig = qconfig
    torch.quantization.prepare_qat(qat_model, inplace=True)

    train_dataset = CustomVOCDataset(
        root='/kaggle/input/voc2012/data',
        year='2012',
        image_set='trainval',
        download=False,
    )
    train_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_train_transforms())

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_dataset = CustomVOCDataset(
        root='/kaggle/input/voc2012/data',
        year='2012',
        image_set='val',
        download=False,
    )
    test_dataset.init_config_yolo(class_mapping=CLASS_MAPPING, custom_transforms=get_valid_transforms())

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    qat_model.train()
    optimizer = optim.Adam(qat_model.parameters(), lr=1e-5)
    yolo_loss_function = YoloLoss()

    NUM_QAT_EPOCHS = 10
    for epoch in range(NUM_QAT_EPOCHS):
        train_fn_qat(train_loader, qat_model, optimizer, yolo_loss_function, epoch)
        test_fn(test_loader, qat_model, yolo_loss_function, epoch)

    qat_model.eval()
    qat_model_cpu = qat_model.to('cpu')
    quantized_final_model = torch.quantization.convert(qat_model_cpu, inplace=False)
    torch.save(quantized_final_model.state_dict(), "/kaggle/working/yolov1_small_quantized_qat_final.pth")
    print("QAT completed and model saved to yolov1_small_quantized_qat_final.pth")


if __name__ == '__main__':
    train()