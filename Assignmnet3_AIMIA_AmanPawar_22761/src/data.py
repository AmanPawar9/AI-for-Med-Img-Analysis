# Getting the Dataset

# Necessary Imports
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
import medmnist
from medmnist import INFO, Evaluator

def get_data(data_flag, BATCH_SIZE = 512):
# data_flag = 'pathmnist'
# data_flag = 'breastmnist'
    download = True

    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[.5], std=[.5])
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    # pil_dataset = DataClass(split='train', download=download)

    # Define the ratio for splitting (e.g., 80% training, 20% validation)
    train_ratio = 0.8
    print(f"Train Ratio : {train_ratio}, Val Ratio: {1.0-train_ratio}")
    # Calculate lengths of train and validation sets based on the ratios
    train_size = int(train_ratio * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # Randomly split the dataset into training and validation
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    #encapsulate data into dataloader form
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training Samples: {len(train_set)}, Validation Samples: {len(val_set)}, Test Samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

    