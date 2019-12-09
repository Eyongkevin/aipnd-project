from torchvision import datasets, transforms
from torch.utils.data import DataLoader

means = [0.485, 0.456, 0.406]
stds =[0.229, 0.224, 0.225]

def get_transforms():
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, stds)])
 
    test_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means,stds)])

    data_transforms = {
        "train":train_transforms, 
        "test":test_transforms, 
        "valid":valid_transforms
    }
    return data_transforms

def get_transformed_data(data_dir:str, data_transforms):
    # Load the datasets with ImageFolder
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    data_transforms = get_transforms()

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])

    image_datasets = {
        "train":train_data, 
        "test": test_data, 
        "valid":valid_data
    }
    return image_datasets


def get_dataloaders(image_datasets):
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    testloader = DataLoader(image_datasets['test'], batch_size=32)
    validloader = DataLoader(image_datasets['valid'], batch_size=32)

    dataloaders = {
        "train":trainloader, 
        "test":testloader, 
        "valid":validloader
    }
    return dataloaders

