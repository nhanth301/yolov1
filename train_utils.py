from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import WIDTH, HEIGHT, DEVICE
from termcolor import colored
from utils import get_bboxes_training, mean_average_precision
import onnxruntime
import torch
import torch.nn as nn


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


def test_fn_onnx(test_loader, onnx_model_path, loss_fn):
    try:
        session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return e

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"ONNX Model '{onnx_model_path}' loaded for testing.")
    print(f"Input Name: {input_name}, Output Name: {output_name}")

    mean_loss = []
    mean_mAP = []

    loop = tqdm(test_loader, desc="Testing ONNX", leave=False)

    for batch_idx, (x, y) in enumerate(loop):
        x_cpu = x.to("cpu") 
        input_np = x_cpu.numpy()

        outputs = session.run([output_name], {input_name: input_np})
        
        out_np = outputs[0]

        out = torch.from_numpy(out_np).to(DEVICE)
    
        loss = loss_fn(out, y)

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.75, box_format="midpoint")

        mean_loss.append(loss.item())
        mean_mAP.append(mAP.item())

        loop.set_postfix(loss=loss.item(), mAP=mAP.item())

    avg_loss = sum(mean_loss) / len(mean_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"ONNX Test \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}", 'cyan')) 


    return avg_mAP

def train_fn_with_mse_distillation(train_loader, student_model, teacher_model, optimizer, yolo_loss_fn, epoch):
    ALPHA = 0.5
    mean_total_loss = []
    mean_yolo_loss = []
    mean_distillation_loss = []
    mean_mAP = []

    total_batches = len(train_loader)
    display_interval = total_batches // 5 

    teacher_model.eval()
    student_model.train()

    distillation_mse_fn = nn.MSELoss(reduction='mean')

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        out = student_model(x)

        with torch.no_grad():
            teacher_out = teacher_model(x)

        yolo_loss = yolo_loss_fn(out, y)
        distillation_loss = distillation_mse_fn(out, teacher_out)

        total_loss = ALPHA * yolo_loss + (1 - ALPHA) * distillation_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pred_boxes, true_boxes = get_bboxes_training(out, y, iou_threshold=0.5, threshold=0.4)
        mAP = mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint")

        mean_total_loss.append(total_loss.item())
        mean_yolo_loss.append(yolo_loss.item())
        mean_distillation_loss.append(distillation_loss.item())
        mean_mAP.append(mAP.item())

        if batch_idx % display_interval == 0 or batch_idx == total_batches - 1:
            print(f"Epoch: {epoch:3} \t Iter: {batch_idx:3}/{total_batches:3} \t Total Loss: {total_loss.item():3.10f} \t YOLO Loss: {yolo_loss.item():3.10f} \t Distill Loss (MSE): {distillation_loss.item():3.10f} \t mAP: {mAP.item():3.10f}")

    avg_total_loss = sum(mean_total_loss) / len(mean_total_loss)
    avg_yolo_loss = sum(mean_yolo_loss) / len(mean_yolo_loss)
    avg_distillation_loss = sum(mean_distillation_loss) / len(mean_distillation_loss)
    avg_mAP = sum(mean_mAP) / len(mean_mAP)
    print(colored(f"Train \t Avg Total Loss: {avg_total_loss:3.10f} \t Avg YOLO Loss: {avg_yolo_loss:3.10f} \t Avg Distill Loss (MSE): {avg_distillation_loss:3.10f} \t Avg mAP: {avg_mAP:3.10f}", 'green'))

    return avg_mAP

def train_fn_qat(train_loader, model, optimizer, yolo_loss_fn, epoch):
    mean_loss = []
    mean_mAP = []
    total_batches = len(train_loader)
    display_interval = total_batches // 5
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x) 
        loss = yolo_loss_fn(out, y)

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
    print(f"\nTrain (QAT) \t loss: {avg_loss:3.10f} \t mAP: {avg_mAP:3.10f}\n")
    return avg_mAP

