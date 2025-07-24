from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import WIDTH, HEIGHT, DEVICE
from termcolor import colored
from utils import get_bboxes_training, mean_average_precision


def get_train_transforms():
    return A.Compose([A.OneOf([A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, val_shift_limit=0.2, p=0.9),
                      A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9)],p=0.9),
                      A.ToGray(p=0.01),
                      A.HorizontalFlip(p=0.2),
                      A.VerticalFlip(p=0.2),
                      A.Resize(height=WIDTH, width=HEIGHT, p=1),
                    #   A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels'])
                      )

def get_valid_transforms():
    return A.Compose([A.Resize(height=WIDTH, width=HEIGHT, p=1.0),
                      ToTensorV2(p=1.0)],
                      p=1.0,
                      bbox_params=A.BboxParams(format='yolo', min_area=0, min_visibility=0, label_fields=['labels']))

def train_fn(train_loader, model, optimizer, loss_fn, epoch):
    mean_loss = []
    mean_mAP = []

    total_batches = len(train_loader)
    display_interval = total_batches // 5 

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

        if batch_idx % display_interval == 0 or batch_idx == total_batches - 1:
            print(f"Epoch: {epoch:3} \t Iter: {batch_idx:3}/{total_batches:3} \t Loss: {loss.item():3.10f} \t mAP: {mAP.item():3.10f}")

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"Train \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}", 'green'))

    return avg_mAP

from tqdm import tqdm
def test_fn(test_loader, model, loss_fn):
    model.eval()
    mean_loss = []
    mean_mAP = []

    loop = tqdm(test_loader, desc="Testing", leave=False)

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.75, box_format="midpoint")

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

        loop.set_postfix(loss=loss.item(), mAP=mAP.item())

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"Test \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}", 'yellow'))

    model.train()

    return avg_mAP