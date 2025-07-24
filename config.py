WIDTH = 448

HEIGHT = 448

LEARNING_RATE = 2e-5

DEVICE = "cpu"

BATCH_SIZE = 8

EPOCHS = 300

NUM_WORKERS = 2

PIN_MEMORY = True

LOAD_MODEL = False

LOAD_MODEL_FILE = "ckpts/final_yolov1.pth.tar"
LOAD_SMALL_MODEL_FILE = "ckpts/KD_yolov1.pth.tar"

CLASS_MAPPING = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}

