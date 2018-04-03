import torch.utils.data as data
from PIL import Image
import os

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def make_dataset(dir):
    images = []

    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    if len(images)>50000:
        return images[:50000]
    else:
        return images


class Dataset(data.Dataset):
    def __init__(self, root, transform=None, loader=pil_loader):
        self.images = make_dataset(root)
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        # TODO
        path = self.images[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img

    def __len__(self):
        return len(self.images)
