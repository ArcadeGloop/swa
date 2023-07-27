# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 20:10:16 2023

@author: trivi
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from PIL import Image
import json
import math
import models
import utils

import tabulate
import itertools
from sklearn.metrics import confusion_matrix




def get_all_models(fpath):
    model_cfg = getattr(models, 'PreResNet14')

    model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    model.cuda()
    
    swa_model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    swa_model.cuda()
    
    our_swa_model = model_cfg.base(*model_cfg.args, num_classes=10, **model_cfg.kwargs)
    our_swa_model.cuda()
    
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1, 
        momentum=0.9, 
        weight_decay=3e-4
    )
    
    
    checkpoint = torch.load(fpath)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    
    
    # original swa
    swa_state_dict = checkpoint['swa_state_dict']
    if swa_state_dict is not None:
        swa_model.load_state_dict(swa_state_dict)
    
    # our swa
    our_swa_state_dict = checkpoint['our_swa_state_dict']
    if our_swa_state_dict is not None:
        our_swa_model.load_state_dict(our_swa_state_dict)
        
    swa_n_ckpt = checkpoint['swa_n']
    if swa_n_ckpt is not None:
        swa_n = swa_n_ckpt
    
    return (model, swa_model, our_swa_model)


def get_cifar_10_image(class_num, iter_num, test_or_train):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    if(test_or_train=="Test"):
        test_or_train = False
    else:
        test_or_train = True
    #transform = transforms.Compose([transforms.ToTensor(), 
    #                                transforms.Normalize((0.5, 0.5, 0.5),
    #                                                     (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root = ".\\data", train = test_or_train, download = True, transform = transform)
    imageset = torchvision.datasets.CIFAR10(root = ".\\data", train = test_or_train, download = True)#, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = False, num_workers = 1)
    
    #classes =["plane","car","bird","cat","deer","dog","frog","horse","ship","truck",]
    dataiter = iter(trainloader)
    imgs, lbls = dataiter.next()
    
    for i in range(100):
        if(lbls[i] == class_num):
            iter_num-=1
            if(iter_num==0):
                return (imgs[i], imageset[i])


def viz_module(imagetensor, image, model, modtype, test_or_train, fname):
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=0., std=1.)
    ])
    
    
    image = image[0]#transforms.ToPILImage(image)#Image.open(str('dog.jpg'))
    image.save("Outputs/"+test_or_train+"_"+fname+".png")
    
    model = model#models.resnet18(pretrained=True)
    print(model)
    
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)
    
    outputs = []
    names = []
    for i, layer in enumerate(conv_layers[0:]):
        image = layer(image)
        outputs.append(image)
        constructed_name = str(layer).split("(")[0] + "-" + str(i+1)+ "\nCin : " + str(layer.in_channels) + "\nCout : " + str(layer.out_channels) + "\nKernel : " + str(layer.kernel_size) + "\nPadding : " + str(layer.padding) + "\nStride : " + str(layer.stride)
        #names.append(str(layer))
        names.append(constructed_name)
    print(len(outputs))
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)
    
    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)
    
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed[i])
        a.axis("off")
        #a.set_title(names[i].split('(')[0], fontsize=30)
        a.set_title(names[i], fontsize=30)
        plt.savefig("Outputs/"+modtype + test_or_train + "_"+ fname +'_feature_maps.jpg', bbox_inches='tight')
    #plt.show()
    
    plt.clf()
    return processed, names



def apply_on_validation_set(model):
    top_n = 3
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    #generator = torch.Generator().manual_seed(args.seed)
    #test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())
    ds = getattr(torchvision.datasets, "CIFAR10")
    path = os.path.join('./data', "cifar10")
    train_transform, test_transform=utils.get_transforms_for("CIFAR10") # our addition
    train_set = ds(root="./data", train=False, download=True, transform=train_transform)
    num_classes = max(train_set.targets) + 1
    generator = torch.Generator().manual_seed(9)
    #train_set, valid_set = torch.utils.data.random_split(train_set, [1-0.2, 0.2]) # our addition
    
    valid_loader = torch.utils.data.DataLoader( # our addition
        train_set,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    with torch.no_grad():
        correct = 0
        total = 0
        top_n_correct = 0
        t_loss = 0
        all_predicted_labels = []
        all_true_labels = []

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            #t_loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            _, top_n_preds = torch.topk(outputs.data, top_n)
            top_n_correct += torch.sum(top_n_preds == labels.unsqueeze(1).expand_as(top_n_preds)).item()

            all_predicted_labels.extend(predicted.tolist())
            all_true_labels.extend(labels.tolist())

    
    
    return (all_true_labels, all_predicted_labels, total, correct, top_n_correct)

def plot_confusion_matrix(cm, classes, fname):
    
    fig,ax = plt.subplots()
    

    cmap = plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("Outputs/"+fname+"Confusion_Matrix.png", bbox_inches = 'tight')
    plt.clf()



def generate_feature_differences(processed_swa, processed_our_swa, names, test_or_train, fname):
    fig = plt.figure(figsize=(30, 50))
    for i in range(len(processed_swa)):
        a = fig.add_subplot(5, 4, i+1)
        imgplot = plt.imshow(processed_our_swa[i] - processed_swa[i])
        a.axis("off")
        #a.set_title(names[i].split('(')[0], fontsize=30)
        a.set_title(names[i], fontsize=30)
        plt.savefig("Outputs/"+ test_or_train + "_"+ fname +'_Difference_feature_maps.jpg', bbox_inches='tight')
    #plt.show()
    
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()



def generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model, test_or_train, fname):
    modtype = "Vanilla_SGD_"
    processed_mod, names_mod = viz_module(imagetensor, image, model, modtype, test_or_train, fname)
    modtype = "SWA_"
    processed_swa, names_swa_mod = viz_module(imagetensor, image, swa_model, modtype, test_or_train, fname)
    modtype = "Proposed_"
    processed_our_swa, names_our_swa_mod = viz_module(imagetensor, image, our_swa_model, modtype, test_or_train, fname)
    
    generate_feature_differences(processed_swa, processed_our_swa, names_mod, test_or_train, fname)
    
    return

def generate_confusion_matrices(model, swa_model, our_swa_model):
    print("done")
    
    print("Creating confusion matrices:")
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


    
    all_true_labels, all_predicted_labels, total, correct, top_n_correct = apply_on_validation_set(model)
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    plot_confusion_matrix(cm, class_names,"Vanilla_SGD_")
    print("this is the confusion matrix")
    all_true_labels, all_predicted_labels, total, correct, top_n_correct = apply_on_validation_set(swa_model)
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    plot_confusion_matrix(cm, class_names,"SWA_")
    print("this is the confusion matrix")
    all_true_labels, all_predicted_labels, total, correct, top_n_correct = apply_on_validation_set(our_swa_model)
    cm = confusion_matrix(all_true_labels, all_predicted_labels)
    plot_confusion_matrix(cm, class_names,"Proposed_")
    
    
    return


def get_fc_weights(model):
    model = model
    
    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    fc_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Linear:
            counter+=1
            model_weights.append(model_children[i].weight)
            fc_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Linear:
                        counter+=1
                        model_weights.append(child.weight)
                        fc_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")
    
    
    return model_weights[0]



def generate_simple_boxplot(class_labels, categories, data_arrays, colors):
    
    
    fig, axs = plt.subplots(2,5, figsize = (15,10))
    
    axs = axs.flatten()
    
    for i, class_label in enumerate(class_labels):
        ax = axs[i]
        
        positions = list(range(1,len(categories)+1))
        
        ax.boxplot([data_arrays[0][i],data_arrays[1][i],data_arrays[2][i]], labels = categories, positions = positions)
        ax.set_title(class_label)
        ax.set_xlabel("Model Type")
        ax.set_ylabel("FC Layer Weights final layer")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig("Outputs/Boxplot_activations_FC_layers.png",bbox_inches="tight")
    
    
    return


def generate_boxplot_of_fc_weights(class_labels, categories, data_arrays, colors):
    """
    Generate a graph with subgraphs containing boxplots for each class.

    Parameters:
        class_labels (list): List of class labels for the subgraphs.
        categories (list): List of category names (e.g., ['Category1', 'Category2', 'Category3']).
        num_samples (int): Number of random data samples per category.

    Returns:
        None (displays the plot using matplotlib)
    """
    # Check if the input data is consistent
    if len(class_labels) != 10 or len(categories) != 3:
        raise ValueError("Number of classes must be 10, and number of categories must be 3.")

    # Create random data for the three categories
    np.random.seed(42)  # For reproducibility
    #data_arrays = []
    

    # Calculate the width of each boxplot
    box_width = 0.25

    # Create a figure and axis objects for the subplots
    fig, axs = plt.subplots(2, 5, figsize=(15, 10), sharey=True)

    # Flatten the axs array to allow easy iteration
    axs = axs.flatten()

    # Iterate over each class label and create boxplots for the categories in each subplot
    for i, class_label in enumerate(class_labels):
        ax = axs[i]

        for j, category in enumerate(categories):
            # Calculate the x position for the current category's boxplot
            x_positions = [1,2,3]#np.arange(len(categories)) + j * box_width

            # Generate the boxplot for the current category's data
            try:
                box_data = data_arrays[j][i]
            except:
                print(j)
                print(i)
            box_props = dict(boxstyle='round', facecolor='lightgray', alpha=0.5)
            bp = ax.boxplot(box_data, positions=x_positions, widths=box_width * 0.7,
                            boxprops=dict(color='black', lw=2), medianprops=dict(color='black', lw=2),
                            whiskerprops=dict(color='black', lw=2), capprops=dict(color='black', lw=2),
                            flierprops=dict(marker='o', markerfacecolor='black', markersize=6, linestyle='none'),
                            patch_artist=True)

            # Set the fill color for the boxplot
            colors = ['red', 'blue', 'green']
            for element in ['boxes']:
                plt.setp(bp[element], color=colors[j], facecolor=colors[j])

        # Set the x-axis labels to the category names
        ax.set_xticks(np.arange(len(categories)) + (len(categories) - 1) * box_width / 2)
        ax.set_xticklabels(categories)
        ax.set_xlabel("Categories")
        ax.set_title(f"Class: {class_label}")

    # Set the y-axis label
    axs[0].set_ylabel("Numerical Values")

    # Adjust layout to avoid overlapping
    plt.tight_layout()

    # Show the plot
    plt.show()
    return

def generate_model_weights_comparison(model, swa_model, our_swa_model):
    
    weights_model = get_fc_weights(model).tolist()
    
    weights_swa = get_fc_weights(swa_model).tolist()
    
    weights_our_swa_model = get_fc_weights(our_swa_model).tolist()
    
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    categories = ['vanilla', 'swa', 'proposed']
    
    data_arrays = [weights_model, weights_swa, weights_our_swa_model]
    
    colors = ['red', 'blue', 'green']
    
    #generate_boxplot_of_fc_weights(class_labels, categories, data_arrays, colors)
    generate_simple_boxplot(class_labels, categories, data_arrays, colors)
    
    
    
    
    return



def main():
    
    #checkpoint = torch.load("./experiment_1_training_checkpoints/training_dir/checkpoint-150.pt")
    
    #modelswa1 = PreResNet14.base()#models.resnet14(pretrained=True)
    #modelourswa1 = PreResNet14.base()#models.resnet14(pretrained=True)
    #modelswa.load_state_dict(checkpoint['swa_state_dict'])
    #modelourswa.load_state_dict(checkpoint['our_swa_state_dict'])
    
    
    print("starting : ")
    #the dog - 
    ###imagetensor, image = get_cifar_10_image(5,2)
    ###imagetensor, image = get_cifar_10_image(2,1)
    ###imagetensor, image = get_cifar_10_image(9,4)
    
    
    #imagetensor, image = get_cifar_10_image(5,7)
    
    #imagetensor, image = get_cifar_10_image(6,1)
    #imagetensor, image = get_cifar_10_image(7,2)
    #imagetensor, image = get_cifar_10_image(8,1)
    
    #imagetensor, image = get_cifar_10_image(4,3)
    #imagetensor, image = get_cifar_10_image(3,3)
    
    #imagetensor, image = get_cifar_10_image(1,1)
    
    
    #model = models.resnet18(pretrained=True)
    
    
    #exponentialPath = "checkpoint_data/trainingcheckpoints_exponential_01/training_dir/checkpoint-200.pt"
    #weighted_average_validation_acc_notscaled = "checkpoint_data/trainingcheckpoints_weightedaverage_validation_accuracy_notscaled/training_dir/checkpoint-200.pt"
    #weighted_average_validation_acc_scaled = "checkpoint_data/trainingcheckpoints_weightedaverage_validation_accuracy_scaled/training_dir/checkpoint-200.pt"
    #weighted_average_validation_loss_notscaled = "checkpoint_data/trainingcheckpoints_weightedaverage_validation_loss_notscaled/training_dir/checkpoint-200.pt"
    
    checkpoint_path = "checkpoint_data_updated/trainingcheckpoints_exponential_01/training_dir/checkpoint-200.pt"
    
    model, swa_model, our_swa_model = get_all_models(checkpoint_path)
    
    #################################
    #abstract away each visualization
    
    
    imagetensor, image = get_cifar_10_image(5,2,"Test")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Test","Dog")
    
    imagetensor, image = get_cifar_10_image(2,1,"Test")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Test","Bird")
    
    imagetensor, image = get_cifar_10_image(9,4,"Test")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Test","Truck")
    
    imagetensor, image = get_cifar_10_image(5,2,"Train")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Train","Dog")
    
    imagetensor, image = get_cifar_10_image(2,1,"Train")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Train","Bird")
    
    imagetensor, image = get_cifar_10_image(9,4,"Train")
    generate_activation_maps(imagetensor, image, model, swa_model, our_swa_model,"Train","Truck")
    
    
    generate_confusion_matrices(model, swa_model, our_swa_model)
    generate_model_weights_comparison(model, swa_model, our_swa_model)
    
    ###############################
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()

