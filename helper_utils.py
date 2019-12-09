from PIL import Image
from load_data_utils import get_transforms
import json
import matplotlib.pyplot as plt

def print_train_config(args):
    # Print all properties prior to training
    print("=============== Training Configuration =============")
    print("Architecture: {}".format(args.arch))
    print("Learning Rate: {}".format(args.learning_rate))
    print("Hidden Layer: {}".format(args.hidden_layer))
    print("Output Layer: {}".format(args.output_layer))
    print("Dropout Prob: {}".format(args.drop_out))
    print("Epochs: {}".format(args.epochs))
    print("Use GPU?: {}".format(args.gpu))
    print("=====================================================")

def str_to_bool(s):
    #Convert string to bool (in argparse context).
    if s.lower() not in ['true', 'false', '1', '0']:
        raise ValueError('Need bool: true, false or 1 and 0; got %r' % s)
    return {'true': True, '1': True, 'false': False, '0':False}[s.lower()]

def process_image(image):
    #Scales, crops, and normalizes a PIL image for a PyTorch model,
    #returns an Numpy array

    # Process a PIL image for use in a PyTorch model
    # Apply the test transform and convert to numpy
    with Image.open(image) as img:  
        data_transforms = get_transforms()
        image = data_transforms['test'](img).numpy()
        
    return image

def load_cat_to_name(file_path):
    # Load dictionary of category to name mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def display_image(image_path,predicted_class):
    # Display image with matplotlib. 
    # This doesn't work on Udacity workspace but will work locally
    fig = plt.figure(figsize=(6,6))
    ax1 = plt.subplot2grid((15,9), (0,0), colspan=9, rowspan=9)
    img = Image.open (image_path)
    ax1.axis('off')
    ax1.set_title(predicted_class)
    ax1.imshow (img)

    
def display_result(class_names, probs):
    # Display results of the probability distribution and class names
    print("***********************************")
    print("PROBABILITY DISTRIBUTION FOR IMAGE CLASSIFICATION")  
    print("***********************************")
    print("\nFLOWER CLASS NAME\t\t\t PROBABILITY DISTRIBUTION")
    for name, prob in zip(class_names, probs):
        print("{}\t\t\t{}".format(name, prob))