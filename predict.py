import argparse
import network_utils
import helper_utils
import torch
import json
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="Predict flower classification with DNN")
    parser.add_argument('input', default='./flowers/test/17/image_03911.jpg', type=str, help="input flower image to predict")
    parser.add_argument('checkpoint', type=str, help='pre-trained model path')
    parser.add_argument('--top_k', default=3, type=int, help='default top_k results')
    parser.add_argument('--category_names', default='./cat_to_name.json', type=str, help='default category file')
    parser.add_argument('--gpu', default='False',type=str, help='If GPU should be enabled')
    return parser.parse_args()


def predict(image, model, use_gpu, topk):
    #Predict the class (or classes) of an image using a trained deep learning model.
    
    if helper_utils.str_to_bool(use_gpu) and torch.cuda.is_available():
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)  # Convert numpy to tensor
        model.cuda()
        print("GPU active")
    else:
        image = torch.from_numpy(image).type(torch.FloatTensor)  # Convert numpy to tensor
        model.cpu()
        print("CPU active: Either cuda is not available or gpu option has been turn off")
    model.eval()  # set model to evaluation mode
                    
    image = torch.unsqueeze(image, dim=0)                    # form a column tensor
    with torch.no_grad ():                                   # Turn off gradient
        output = model.forward(image)
    preds, classes = torch.exp(output).topk(topk)            # Get prediction and classes of top 5
    probs = preds.cpu().numpy().tolist()[0]
    classes = classes.cpu().numpy().tolist()[0]
    
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    topk_classes = [idx_to_class[i] for i in classes]
    
    return probs, topk_classes


    
def main():
    args = get_args()
    processed_image = helper_utils.process_image(args.input) # Process the image to numpy array
    model = network_utils.loading_model(args.checkpoint)

    probs, topk_classes = predict(processed_image, model, args.gpu, args.top_k)
    cat_to_name = helper_utils.load_cat_to_name(args.category_names)
    class_names = [cat_to_name [item] for item in topk_classes]
    max_prob_idx = np.argmax(probs)
    max_class_nb = topk_classes[max_prob_idx]
    predicted_class = cat_to_name[max_class_nb]
    #helper_utils.display_image(args.input,predicted_class)

    helper_utils.display_result(class_names, probs)

if __name__ == '__main__':
    main()