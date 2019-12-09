import torch
import argparse
import load_data_utils
import network_utils
import helper_utils


def get_args():
    # Use argparse to get argument from terminal
    parser = argparse.ArgumentParser(description="Training a model for flower classification")
    parser.add_argument('--data_dir', default='./flowers', type=str, help="data directory")
    parser.add_argument('--save_dir', default='./', type=str, help="directory to save checkpoints")
    parser.add_argument('--arch', default='alexnet', help='DNN models to use like: alexnet, vgg13')
    parser.add_argument('--hidden_layer', default=512, type=int, help='number of neurons in hidden layer')
    parser.add_argument('--output_layer', default=102, type=int, help='number of output layers')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--drop_out', default=0.3, type=float, help='dropout probability')
    parser.add_argument('--epochs', default=7, type=int, help='number of epochs for training')
    parser.add_argument('--gpu', default="False", type=str, help='If GPU should be enabled')
    parser.add_argument('--print_every', default=10, type=int, help='Set number of steps after every print')
    return parser.parse_args()

def validation(model, validloader, criterion, device):
    # function for the validation pass
    valid_loss = 0
    accuracy = 0
    for images, labels in validloader:
        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy


def train(model, dataloaders, criterion, optimizer, epochs, print_every, use_gpu ):
    #Train a model with a pre-trained network
    steps = 0
    running_loss = 0

    if helper_utils.str_to_bool(use_gpu) and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU active")
    else:
        device = torch.device("cpu")
        print("CPU active: Either cuda is not available or gpu option has been turn off")

    model.to(device)  # Set model to GPU or CPU mood base on availability

    for e in range(epochs):
        model.train()
        for images, labels in dataloaders['train']:
            images, labels = images.to(device), labels.to(device)
            steps += 1
            
            
            optimizer.zero_grad()  # zero the gradients
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()     # back propagate the loss
            optimizer.step()    # Update the weights
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()
                
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloaders['valid'], criterion, device)
                    
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Valid Loss: {:.3f}.. ".format(valid_loss/len( dataloaders['valid'])),
                    "Valid Accuracy: {:.3f}".format(accuracy/len( dataloaders['valid'])))
                
                running_loss = 0
                
                # Make sure training is back on
                model.train()


def main():
    args = get_args()
    helper_utils.print_train_config(args)
    transforms = load_data_utils.get_transforms()
    image_datasets = load_data_utils.get_transformed_data(args.data_dir, transforms)
    dataloaders = load_data_utils.get_dataloaders(image_datasets)

    model, input_layer = network_utils.build_network(args.arch, args.hidden_layer,args.output_layer, args.drop_out)
    
    class_to_index = image_datasets['train'].class_to_idx
    criterion = network_utils.get_loss_func()
    classifier = network_utils.get_classifier(model)
    optimizer = network_utils.get_optim(classifier.parameters(), args.learning_rate)
    train(model, dataloaders, criterion, optimizer, args.epochs, args.print_every, args.gpu)
    network_utils.save_model(model, args.arch, args.save_dir, args.epochs, args.learning_rate,
                                input_layer, args.hidden_layer, args.output_layer,
                                args.drop_out, class_to_index, classifier)



if __name__ == '__main__':
    main()