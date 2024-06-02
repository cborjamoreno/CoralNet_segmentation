import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as transforms
import numpy as np

from PIL import Image
from pathlib import Path

class Cityscapes(torchvision.datasets.Cityscapes):
    def __init__(self, *arg, use_train_classes=False, **kwargs):
        self.use_train_classes = use_train_classes

        super().__init__(*arg, **kwargs)

        if self.use_train_classes:
            self.num_classes = 19
            self.ignore_index = 255
        else:
            self.num_classes = len(self.classes) - 2 # discard -1 and 0 classes
            self.ignore_index = 0

        self.train_labels_map = {c.train_id: c.id for c in self.classes}
        self.train_labels_map[255] = 0

    def from_train_labels(self, labels: torch.Tensor) -> torch.Tensor:
        return labels.apply_(self.train_labels_map.get)

    def color_labels(self, labels: torch.Tensor) -> np.array:
        result = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
        for c in self.classes:
            if c.train_id == self.ignore_index:
                continue
            result[labels == c.train_id] = c.color
        return result

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == 'semantic':
            if self.use_train_classes:
                return f'{mode}_labelTrainIds.png'
            else:
                return f'{mode}_labelIds.png'
            
class SEBENS(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.num_classes = 61
        self.ignore_index = 0

        if self.train:
            self.image_paths = sorted(list((self.root / 'images' / 'train').glob('*.png')) + list((self.root / 'images' / 'train').glob('*.jpg')))
            self.label_paths = sorted(list((self.root / 'labels' / 'train').glob('*.png')) + list((self.root / 'labels' / 'train').glob('*.jpg')))
        else:
            self.image_paths = sorted(list((self.root / 'images' / 'test').glob('*.png')) + list((self.root / 'images' / 'test').glob('*.jpg')))
            self.label_paths = sorted(list((self.root / 'labels' / 'test').glob('*.png')) + list((self.root / 'labels' / 'test').glob('*.jpg')))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # assuming labels are grayscale images

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Convert label to a LongTensor
        label = torch.from_numpy(np.array(label)).long()

        return image, label
    
class CORALES(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.num_classes = 57
        self.ignore_index = 0

        if self.train:
            self.image_paths = sorted(list((self.root / 'images' / 'train').glob('*.png')) + list((self.root / 'images' / 'train').glob('*.jpg')))
            self.label_paths = sorted(list((self.root / 'labels' / 'train').glob('*.png')) + list((self.root / 'labels' / 'train').glob('*.jpg')))
        else:
            self.image_paths = sorted(list((self.root / 'images' / 'test').glob('*.png')) + list((self.root / 'images' / 'test').glob('*.jpg')))
            self.label_paths = sorted(list((self.root / 'labels' / 'test').glob('*.png')) + list((self.root / 'labels' / 'test').glob('*.jpg')))

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')  # assuming labels are grayscale images

        # Convert label to a numpy array
        label_np = np.array(label)

        # Replace 255 with 0
        label_np[label_np == 255] = 0

        # Convert label back to an Image
        label = Image.fromarray(label_np)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # Convert label to a LongTensor
        label = torch.from_numpy(np.array(label)).long()
        
        return image, label
    