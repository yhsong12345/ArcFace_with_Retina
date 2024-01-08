import torch
from torch.utils.data import DataLoader, Subset
from data.preprocess import WiderFaceDetection, detection_collate




# data constants
VALID_SPLIT = .15
NUM_WORKERS = 0






def create_datasets(s, f):
    img_dim =s
    path = f
    training_dataset = path + '/train/label.txt'
    train_dataset = WiderFaceDetection(training_dataset, preproc=img_dim)
    train_dataset_size = len(train_dataset)
    # validation size = number of validation images
    valid_size = int(VALID_SPLIT*train_dataset_size)
    # all the indices from the training set
    indices = torch.randperm(len(train_dataset)).tolist()
    # final train dataset discarding the indices belonging to `valid_size` and after
    dataset_train = Subset(train_dataset, indices[:-valid_size])
    # final valid dataset from indices belonging to `valid_size` and after
    dataset_valid = Subset(train_dataset, indices[-valid_size:])
    print(f"Total training images: {len(dataset_train)}")
    print(f"Total validation images: {len(dataset_valid)}")

    return train_dataset, dataset_valid,



def create_data_loaders(dataset_train, dataset_valid, BATCH_SIZE):
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn = detection_collate
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn = detection_collate
    )
    return train_loader, valid_loader


