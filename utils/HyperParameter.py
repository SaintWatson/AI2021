from torch._C import import_ir_module_from_buffer
from torch.cuda import is_available
from torchvision import transforms
from torchvision.transforms.functional import resize
from torchvision.transforms.transforms import Grayscale, RandomHorizontalFlip, RandomResizedCrop, RandomRotation, RandomVerticalFlip, Resize

IMG_SIZE = 256
DEVICE = "cuda:5" if is_available()  else "cpu"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_EPOCH = 10
VAL_RATIO = 0.1

TRAIN_TFM = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
])

TEST_TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])