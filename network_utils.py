from torchvision import models
from collections import OrderedDict
from torch import nn
import torch
import os
import helper_utils

def build_network(arch, hidden_layer, output_layer, drop_out):
    # Build network from pre-trained model.
    # Allow the user to load at least one more model architecture
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        classifier_input_size = model.classifier[1].in_features
    elif arch == 'vgg19':
        model = models.vgg16(pretrained=True)
        classifier_input_size = model.classifier[0].in_features
    elif arch == 'resnet101':
        model = models.resnet101(pretrained=True)
        classifier_input_size = model.fc.in_features
    else:
        raise RuntimeError("invalid model: Please chose among `alexnet`, `vgg19` and `resnet101`")

   # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    #input_layer = get_classifier_input_layer(model)
   
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(classifier_input_size, hidden_layer)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=drop_out)),
                            ('fc2', nn.Linear(hidden_layer, output_layer)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
        
    set_classifier(model, classifier)
    return model, classifier_input_size


def get_loss_func():
    # Get loss function
    return nn.NLLLoss()
def get_optim(params, lr):
    # Get optimizer 
    return torch.optim.Adam(params, lr)

def get_classifier_input_layer(model):
    # Get classifier or fc property of a model
    # Some models like alexnet have 'classifier' but models like resnet18 have 'fc'
    # Also, for models with classifier property, some start with a dropout. In this case, we get the in_feature of 
    # the second index which will represent the input layer
    try:
        input_layer = model.fc.in_features
    except AttributeError:
        try:
            input_layer = model.classifier[0].in_features
        except AttributeError:
            input_layer = model.classifier[1].in_features
        
    return input_layer
    
def get_classifier(model):
    # Get classifier or fc base on model types. 
    try:
        if model.fc:
            classifier = model.fc
    except AttributeError:
        classifier = model.classifier
    return classifier

def set_classifier(model, classifier):
    # Set classifier or fc base on model type
    try:
        if model.fc:
            model.fc = classifier
    except AttributeError:
        model.classifier = classifier

def save_model(model, arch, save_dir, epochs, lr,
                 input_layer, hidden_layer, output_layer, drop_out, class_to_idx, classifier):
    # Save trained model
    # Here, I save a lot of properties in case the model needs to be continue training
    if not os.path.isdir(save_dir):
        print("--save_dir not a directory. Saving to root directory")
        save_dir = './'

    save_path = os.path.join(save_dir, "checkpoint_{}.pth".format( arch)) 
    model.cpu()

    checkpoint = {
        'arch': arch,
        'epoch': epochs,
        'lr':lr,
        'drop_out':drop_out,
        'input_layer':input_layer,
        'hidden_layer': hidden_layer,
        'output_layer': output_layer,
        'classifier': classifier,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint,save_path)
    print("Successfully saved model checkpoint at {}".format(save_path))


def loading_model (checkpoint_path):
    # Load model from checkpoint for prediction or continual training
    checkpoint = torch.load (checkpoint_path) 
    model = getattr(models, checkpoint['arch'])(pretrained=True)
        
    set_classifier(model, checkpoint ['classifier'])
    model.load_state_dict (checkpoint ['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']
    
    return model