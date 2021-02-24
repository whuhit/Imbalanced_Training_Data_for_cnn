"""
Cifar10 Dataloader implementation, used in CondenseNet
"""
import logging
import torchvision.transforms as v_transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


class CIFAR10(Dataset):
    def __init__(self, data_dir, data_file, transform=None):
        self.data_dir = data_dir
        self.data_file = data_file
        self.transform = transform
        self.ids = open(self.data_file).read().splitlines()
        print(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        name, label = self.ids[i].split()
        img_file = os.path.join(self.data_dir, name)
        label = int(label)

        image = Image.open(img_file)
        if self.transform:
            image = self.transform(image)
        return image, label


class Cifar10DataLoader:
    def __init__(self, config):
        self.logger = logging.getLogger("Cifar10DataLoader")
        self.logger.info("Loading DATA.....")
        normalize = v_transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                           std=[0.2471, 0.2435, 0.2616])

        train_set = CIFAR10(config.train_folder,
                            data_file=config.train_file,
                            transform=v_transforms.Compose([
                                v_transforms.RandomCrop(32, padding=4),
                                v_transforms.RandomHorizontalFlip(),
                                v_transforms.ToTensor(),
                                normalize,
                            ]))
        valid_set = CIFAR10(config.test_folder,
                            data_file=config.test_file,
                            transform=v_transforms.Compose([
                                v_transforms.ToTensor(),
                                normalize,
                            ]))

        self.train_loader = DataLoader(
            train_set, batch_size=config.batch_size, shuffle=True)
        self.val_loader = DataLoader(
            valid_set,
            batch_size=config.batch_size,
            shuffle=False)

        self.logger.info(
            f"nums of train DATA: {len(train_set)}, nums of val DATA: {len(valid_set)}")


if __name__ == '__main__':
    import numpy as np
    from easyPlog import Plog

    nums_array = np.array([[328, 328, 328, 328, 328, 328, 328, 328, 328, 328],
              [262, 262, 262, 262, 262, 394, 394, 394, 394, 394],
              [197, 197, 197, 197, 197, 459, 459, 459, 459, 459],
              [402, 320, 320, 320, 320, 320, 320, 320, 320, 320],
              [476, 312, 312, 312, 312, 312, 312, 312, 312, 312],
              [239, 338, 338, 338, 338, 338, 338, 338, 338, 338],
              [210, 341, 341, 341, 341, 341, 341, 341, 341, 341],
              [270, 283, 296, 309, 321, 335, 344, 361, 373, 386],
              [249, 252, 259, 272, 290, 314, 345, 384, 430, 485],
              [410, 410, 273, 273, 273, 273, 273, 273, 410, 410],
              [500, 500, 214, 214, 214, 214, 214, 214, 500, 500]])

    for i in range(1, 12):
        log = Plog(f"../assets/data/dla_cifar/cifar10_small/dist_{i}.txt", cover=True)
        count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for line in open("../assets/data/dla_cifar/dist_default/train.txt"):
            label = int(line.split()[1])
            count[label] += 1
            if count[label] > nums_array[i - 1][label]:
                continue
            else:
                log.log(line.strip())

    # 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12, 6, 6, 6, 6, 6, 14, 14, 14, 14, 14, 12.25, 9.75, 9.75, 9.75, 9.75, 9.75, 9.75, 9.75, 9.75, 9.75, 14.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 7.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 10.3, 6.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 8.24, 8.63, 9.02, 9.41, 9.80, 10.20, 10.5, 10.99, 11.37, 11.76, 7.58, 7.68, 7.91, 8.29, 8.83, 9.57, 10.52, 11.70, 13.11, 14.79, 12.5, 12.5, 8.33, 8.33, 8.33, 8.33, 8.33, 8.33, 12.5, 12.5, 15.22, 15.22, 6.52, 6.52, 6.52, 6.52, 6.52, 6.52, 15.22, 15.22
