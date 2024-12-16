import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# As mentioned in readme.html, each of these files is a Python "pickled" object produced with cPickle,
# so, we "unpickle" them accodringly.
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')  # Specify 'latin1' to avoid decoding issues : Recommended by ChatGPT.
    return dict

# Load all data batches and limits the dataset for execution speed if the limit value is set to one higher than 0.
def load_data(data_dir, limit):
    # Axis array initialisation.
    x_train = []
    y_train = []    

    # CIFAR-10 has 5 training batches and 1 test batch
    for i in range(1, 6):
        data_dict = unpickle(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(data_dict['data'])
        y_train += data_dict['labels']
    
    x_train = np.concatenate(x_train) # Training data
    y_train = np.array(y_train) # Training labels
    
    # Load test batch
    test_dict = unpickle(os.path.join(data_dir, 'test_batch'))
    x_test = test_dict['data'] # Testing data
    y_test = np.array(test_dict['labels']) # Testing labels

    # If limit is 0, it will be interpreted as false and not limit the dataset,
    # if limit id higher than 0, it will limit the dataset acccordingly.
    if limit>0:
        olen=len(x_train)
        x_train, y_train, x_test, y_test = limit_dataset(limit, x_train, y_train, x_test, y_test)
        print("Limiting dataset from ",olen," to ", len(x_train))
    
    return x_train, y_train, x_test, y_test

# This function is mainly for test purposes; It limits the training datasets to a set number (num) and the testing
# datasets accordingly as in the database for every 5 training images there is 1 for testing.
def limit_dataset(num, x_train, y_train, x_test, y_test):
    x_train, y_train = x_train[:num], y_train[:num]  
    x_test, y_test = x_test[:num//5], y_test[:num//5]
    return x_train, y_train, x_test, y_test

def reshape(x_train, x_test):
    # Reshape data to [num_samples, channels, height, width]
    x_train = x_train.reshape(-1, 3, 32, 32).astype('float32')  # [50000, 3, 32, 32]
    x_test = x_test.reshape(-1, 3, 32, 32).astype('float32')    # [10000, 3, 32, 32]

    # Normalize data to range [-1, 1]
    x_train = (x_train / 255.0) * 2 - 1
    x_test = (x_test / 255.0) * 2 - 1
    return x_train, x_test

class CIFAR10Dataset(Dataset):
    def __init__(self, data, labels, transform=None, normalize=None):
        self.data = data
        self.labels = labels
        self.transform = transform  # Augmentation transforms
        self.normalize = normalize 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Reshape the image from [3072] to [3, 32, 32]
        image = self.data[idx].reshape(3, 32, 32).astype('float32')  # NumPy array
        label = self.labels[idx]

        # Conversion to PIL image : Recommended by ChatGPT
        image = to_pil_image(torch.tensor(image))

        # If the data is to be augmented, do the augmentation
        if self.transform:
            image = self.transform(image)

        image = transforms.ToTensor()(image)

        # If the data is to be normalized, do it
        if self.normalize:
            image = self.normalize(image)

        return image, torch.tensor(label, dtype=torch.long)


    
def data_loader(batch):
    import torchvision.transforms as transforms
    x_train, y_train, x_test, y_test = load_data("DB", 0)

    # Data augmentation via transform
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10) #+- 10 degrees of ratation
    ])

    normalize = transforms.Normalize((0.5,), (0.5,))

    train_dataset = CIFAR10Dataset(x_train, y_train, transform=train_transform, normalize=normalize)
    test_dataset = CIFAR10Dataset(x_test, y_test, normalize=normalize)  # No augmentation for test set

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    return train_loader, test_loader

