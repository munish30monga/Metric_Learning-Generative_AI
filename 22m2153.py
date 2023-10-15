# EE 782: Advanced Topics in Machine Learning
# Assignment 2: Metric Learning and Generative AI
# NAME: Munish Monga
# ROLL NO: 22M2153
# Github Repo Link: https://github.com/munish30monga/Metric_Learning-Generative_AI

import pathlib as pl                                                # for path handling
import matplotlib.pyplot as plt                                     # for plotting
from PIL import Image                                               # for image handling
from sklearn.model_selection import train_test_split                # for splitting data
import random                                                       # for random number generation
import torch                                                        # for deep learning functionality
import numpy as np                                                  # for numerical operations  
import matplotlib.pyplot as plt                                     # for plotting 
import torch.nn as nn                                               # for neural network functionality
import torch.nn.functional as F                                     # for neural network functionality        
import torch                                                        # for deep learning functionality
from torchvision import transforms                                  # for image transformations
from torchvision.transforms import CenterCrop                       # for cropping images
from torch.utils.data import Dataset, DataLoader                    # for dataset handling
from tqdm import tqdm                                               # for progress bar
import timm                                                         # for pretrained models
import albumentations as A                                          # for image augmentations
from albumentations.pytorch import ToTensorV2                       # for image augmentations
from itertools import combinations                                  # for combinations                                         # for sampling
from prettytable import PrettyTable                                 # for table formatting
import copy                                                         # for copying objects   
import argparse                                                     # for command line arguments
from sklearn.metrics import classification_report                   # for model evaluation
import shutil                                                       # for file handling
import torchvision.transforms as T                                  # for image transformations
import os                                                           # for file handling
from torchvision.utils import save_image                            # for saving images
from IPython.display import Image as IPImage, HTML                  # for displaying images

# SET CUDA DEVICE
os.environ['CUDA_VISIBLE_DEVICES'] = '4' 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def visualize_pairs(X, Y):
    """
    Visualizes pairs of images from X along with their label from Y.

    Args:
    - X (list): List of image pairs.
    - Y (list): List of labels.
    """
    fig, axes = plt.subplots(4, 4, figsize=(15, 5*4))
    
    pos_idx = [i for i, label in enumerate(Y) if label == 1.0]
    neg_idx = [i for i, label in enumerate(Y) if label == 0.0]

    # Shuffle the indices
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)

    for row in range(4):
        # Display positive pairs
        img_path1, img_path2 = X[pos_idx[row]]
        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)
        person1_name = img_path1.parent.name  # Assuming the parent directory is the person's name
        person2_name = img_path2.parent.name
        
        axes[row, 0].imshow(np.array(image1), cmap='gray')
        axes[row, 0].axis('off')
        axes[row, 0].set_title(person1_name, fontsize = 18)

        axes[row, 1].imshow(np.array(image2), cmap='gray')
        axes[row, 1].axis('off')
        axes[row, 1].set_title(person2_name, fontsize = 18)

        # Display negative pairs
        img_path1, img_path2 = X[neg_idx[row]]
        image1 = Image.open(img_path1)
        image2 = Image.open(img_path2)
        person1_name = img_path1.parent.name  # Assuming the parent directory is the person's name
        person2_name = img_path2.parent.name

        axes[row, 2].imshow(np.array(image1), cmap='gray')
        axes[row, 2].axis('off')
        axes[row, 2].set_title(person1_name, fontsize = 18)

        axes[row, 3].imshow(np.array(image2), cmap='gray')
        axes[row, 3].axis('off')
        axes[row, 3].set_title(person2_name, fontsize = 18)

    # Setting main titles for Positive and Negative pairs
    fig.text(0.25, 1, 'Positive Pairs', ha='center', va='center', fontsize=24)
    fig.text(0.75, 1, 'Negative Pairs', ha='center', va='center', fontsize=24)
    
    plt.tight_layout()
    plt.show()
    
def split_data(persons_dict):
    """
    Split persons data for training a Siamese network.
    
    Args:
    - persons_dict (dict): Dictionary with person names as keys and the number of images as values.

    Returns:
    - train_persons_dict (dict): Dictionary for training set.
    - valid_persons_dict (dict): Dictionary for validation set.
    - test_persons_dict (dict): Dictionary for test set.
    """
    single_image_persons = [person for person, count in persons_dict.items() if count == 1]
    multi_image_persons = [person for person, count in persons_dict.items() if count > 1]

    # Split the multi-image persons into 80% for training and 20% for validation + test
    train_multi_persons, remaining_persons = train_test_split(multi_image_persons, test_size=0.2, random_state=42)
    
    # Split the remaining 20% equally into validation and test sets
    valid_persons, test_persons = train_test_split(remaining_persons, test_size=0.5, random_state=42)
    
    # Add single image persons to training set
    train_persons = single_image_persons + train_multi_persons
    
    # Create dictionaries for each set
    train_persons_dict = {person: persons_dict[person] for person in train_persons}
    valid_persons_dict = {person: persons_dict[person] for person in valid_persons}
    test_persons_dict = {person: persons_dict[person] for person in test_persons}
    
    return train_persons_dict, valid_persons_dict, test_persons_dict

def generate_positive_pairs(persons_dict, num_augmentations=5):
    transforms = A.Compose([
        # A.Rotate(limit=15),
        A.ColorJitter(brightness=0.5, contrast=0.5),
        A.HorizontalFlip(),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        # A.RandomRotate90(),
        # A.ToGray(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=25, max_width=25, fill_value=0, p=0.5),
        ToTensorV2()
    ])
    data_dir = pl.Path('./dataset/lfw/')
    augmented_dataset_dir = pl.Path('./augmented_dataset/')
    
    # Remove the existing directory and its contents
    if augmented_dataset_dir.exists():
        shutil.rmtree(augmented_dataset_dir)
    
    # Recreate the directory
    augmented_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    person_names = list(persons_dict.keys())
    augmented_persons_count = 0
    total_augmented_images = 0
    
    for idx, person in enumerate(person_names):
        person_dir = data_dir / person
        images = list(person_dir.glob("*.jpg"))
        
        if len(images) == 1:
            augmented_persons_count += 1
            person_augmented_dir = augmented_dataset_dir / person
            person_augmented_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(num_augmentations):
                augmented_image_tensor = transforms(image=np.array(Image.open(images[0])))["image"]
                # Convert tensor back to PIL Image
                augmented_image = T.ToPILImage()(augmented_image_tensor)
                image_path = person_augmented_dir / f"augmented_{i}.jpg"
                augmented_image.save(image_path)
                total_augmented_images += 1

    print(f"Total number of augmented images generated = Augmented Persons x Augmentations per person = {augmented_persons_count} x {num_augmentations} = {total_augmented_images}")
    
def generate_pairs(persons_dict, max_positive_combinations, apply_augmentation, num_augmentations):
    """
    Generates pairs of images (both positive and negative) for Siamese training.
    
    Args:
    - persons_dict (dict): Dictionary with person names as keys and the number of images as values.
    - max_positive_combinations (int): Maximum number of positive combinations per person.
    Returns:
    - X (list): List of image pairs.
    - Y (list): List of labels indicating whether the pair is positive or negative.
    - positive_pairs_count (int): Number of positive pairs.
    - negative_pairs_count (int): Number of negative pairs.
    """
    X, Y = [], []
    person_names = list(persons_dict.keys())
    
    data_dir = pl.Path('./dataset/lfw/')
    augmented_data_dir = pl.Path('./augmented_dataset/')
    
    positive_pairs_count, negative_pairs_count = 0, 0
            
    for idx, person in enumerate(person_names):
        person_dir = data_dir / person
        images = list(person_dir.glob("*.jpg"))
            
        # Generate positive pairs for original images
        if len(images) >= 2:
            all_combinations = list(combinations(images, 2))
            selected_combinations = random.sample(all_combinations, min(max_positive_combinations, len(all_combinations)))
            
            for img1, img2 in selected_combinations:
                X.append([img1, img2])
                Y.append(1.0)  # Same person
                positive_pairs_count += 1

                # Generate a negative pair for every positive pair
                other_persons = [p for p in person_names if p != person]
                if other_persons:
                    other_person = random.choice(other_persons)
                    other_person_dir = data_dir / other_person
                    other_person_images = list(other_person_dir.glob("*.jpg"))
                    
                    if other_person_images:
                        img1_neg = random.choice(images)
                        img2_neg = random.choice(other_person_images)
                        X.append([img1_neg, img2_neg])
                        Y.append(0.0)  # Different persons
                        negative_pairs_count += 1
                        
        # Generate positive pairs for augmented images
        if len(images) == 1 and apply_augmentation:
            person_augmented_dir = augmented_data_dir / person
            augmented_images = list(person_augmented_dir.glob("*.jpg"))
            
            for img1 in images:  # This will loop only once since len(images) == 1
                for img2 in augmented_images:
                    X.append([img1, img2])  # Pairing original image with each augmented image
                    Y.append(1.0)  # Same person using augmented images
                    positive_pairs_count += 1

                    # Generate a negative pair for every positive pair
                    other_persons = [p for p in person_names if p != person]
                    if other_persons:
                        other_person = random.choice(other_persons)
                        other_person_dir = data_dir / other_person
                        other_person_images = list(other_person_dir.glob("*.jpg"))

                        if other_person_images:
                            img1_neg = img1
                            img2_neg = random.choice(other_person_images)
                            X.append([img1_neg, img2_neg])
                            Y.append(0.0)  # Different persons
                            negative_pairs_count += 1


    return X, Y, positive_pairs_count, negative_pairs_count

def preprocess_data(X):
    """
    Preprocess the images in X using albumentations by applying a center crop and normalization.

    Args:
    - X (list): List of image pairs.

    Returns:
    - X_processed (list): List of preprocessed image pairs.
    """
    transform = A.Compose([
        A.CenterCrop(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    X_processed = []

    for pair in X:
        processed_pair = [transform(image=np.array(Image.open(img_path)))["image"] for img_path in pair]
        X_processed.append(processed_pair)

    return X_processed

def dict_to_tensors(persons_dict, max_positive_combinations, apply_augmentation, num_augmentations):
    """
    Converts a dictionary of persons into tensors of image pairs and labels.

    Args:
    - persons_dict (dict): Dictionary with person names as keys and the number of images as values.

    Returns:
    - X_tensor (torch.Tensor): Tensor of image pairs.
    - Y_tensor (torch.Tensor): Tensor of labels.
    """
    X, Y, _, _ = generate_pairs(persons_dict, max_positive_combinations, apply_augmentation=apply_augmentation, num_augmentations=num_augmentations)    
    
    # Preprocess the data (apply center cropping and normalization)
    X = preprocess_data(X)

    # Assuming all images have the same dimensions after cropping and transformations
    first_image = X[0][0]
    _, img_height, img_width = first_image.size()

    # Convert the images to tensors
    X_tensor = torch.empty((len(X), 2, 3, img_height, img_width))  # Shape: (num_pairs, 2, 3, img_height, img_width)
    
    for i, (img1, img2) in enumerate(X):
        X_tensor[i, 0] = img1
        X_tensor[i, 1] = img2
    
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    return X_tensor, Y_tensor

class SiameseDataset(Dataset):
    """
    Dataset class for Siamese networks.

    Attributes:
    - X (list): List of image pairs.
    - Y (list): List of labels indicating whether the pair is positive or negative.

    Methods:
    - __len__(): Returns the number of pairs in the dataset.
    - __getitem__(index): Returns the image pair and label at the given index.
    """
    def __init__(self, X, Y):
        """
        Initializes the SiameseDataset with given image pairs and labels.

        Args:
        - X (list): List of image pairs.
        - Y (list): List of labels.
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        img1 = self.X[index, 0]
        img2 = self.X[index, 1]
        label = self.Y[index]
        return (img1, img2), label

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor.

    Args:
    - tensor (torch.Tensor): The normalized tensor.
    - mean (list): The mean values used for normalization.
    - std (list): The standard deviation values used for normalization.

    Returns:
    - torch.Tensor: The denormalized tensor.
    """
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    return tensor * std + mean

def visualize_batch(dataloader, num_samples=4):
    # Fetch one batch of data
    (img1_batch, img2_batch), labels_batch = next(iter(dataloader))
    
    # Denormalize the images
    img1_batch = denormalize(img1_batch)
    img2_batch = denormalize(img2_batch)
    
    # Randomly sample indices for visualization
    indices = random.sample(range(len(labels_batch)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 2 * num_samples))
    
    for i, idx in enumerate(indices):
        img1 = np.clip(img1_batch[idx].permute(1, 2, 0).numpy(), 0, 1)
        img2 = np.clip(img2_batch[idx].permute(1, 2, 0).numpy(), 0, 1)

        
        axes[i, 0].imshow(img1, cmap='gray')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img2, cmap='gray')
        axes[i, 1].axis('off')
        
        if labels_batch[idx] == 1.0:
            label = "Positive Pair"
        else:
            label = "Negative Pair"
        
        # Place the label to the left of the images in vertical fashion
        fig.text(0.1, (num_samples - i - 0.5) / num_samples, label, ha='center', va='center', fontsize=14, rotation=90)
    
    plt.tight_layout()
    plt.show()
class SiameseNetwork(nn.Module):
    """
    Siamese Neural Network for learning embeddings using pairs of images. 

    Attributes:
    - base_model (nn.Module): The base model used for feature extraction.
    - embedding_size (int): Size of the embedding produced by the base model.
    - projection (nn.Linear): Linear layer to project embeddings to desired size (512).

    Methods:
    - forward_one(x): Computes the embedding for a single input image.
    - forward(input1, input2): Computes the embeddings for a pair of input images.
    """
    def __init__(self, base_model, unfreeze_last_n):
        """
        Initializes the SiameseNetwork with a given base model.

        Args:
        - base_model (str): Name of the base model to use
        - unfreeze_last_n (int): Number of layers to unfreeze from the end
        """
        super(SiameseNetwork, self).__init__()
        
        # Create the model
        self.base_model = timm.create_model(base_model, pretrained=True)
        
        # Get the number of features (embedding size) from the base model
        self.embedding_size = self.base_model.num_features
        
        # Remove the classification head
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        # If unfreeze_last_n is -1, make all layers trainable
        if unfreeze_last_n == -1:
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers initially
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze the last n layers
            num_layers = len(list(self.base_model.children()))
            for i, child in enumerate(self.base_model.children()):
                if i >= num_layers - unfreeze_last_n:
                    for param in child.parameters():
                        param.requires_grad = True
        
        # Projection layer to get embeddings of size 512
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_size, 512),
            nn.BatchNorm1d(512))

    def forward_once(self, x):
        # Forward pass for one input
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x

    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese networks.

    Attributes:
    - margin (float): Margin for the contrastive loss.

    Methods:
    - forward(output1, output2, label): Computes the contrastive loss given a pair of embeddings and their label.
    """
    def __init__(self, margin=2.0):
        """
        Initializes the ContrastiveLoss with a given margin.

        Args:
        - margin (float): Margin for the contrastive loss.
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

def choose_loss_function(loss_function, margin):
    """
    Chooses the loss function based on the given name.

    Args:
    - loss_function (str): Name of the loss function. Options are "BCE", "hinge_loss", and "contrastive".

    Returns:
    - loss_function (nn.Module): The loss function.
    """
    if loss_function == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using Cross Entropy Loss...")
    elif loss_function == "hinge_loss":
        criterion = torch.nn.MarginRankingLoss(margin=margin)
        print(f"Using Hinge Loss with margin={margin}...")
    elif loss_function == "contrastive":
        criterion = ContrastiveLoss(margin=margin)
        print(f"Using Contrastive Loss with margin={margin}...")
    else:
        raise ValueError("Invalid loss_function!")
    return criterion

def choose_optimizer(model, optimizer_type, learning_rate, weight_decay):
    if optimizer_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            print(f"Using Adam optimizer with lr={learning_rate}, weight_decay={weight_decay}...")
    elif optimizer_type == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print(f"Using Adagrad optimizer with lr={learning_rate}, weight_decay={weight_decay}...")
    elif optimizer_type == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print(f"Using RMSprop optimizer with lr={learning_rate}, weight_decay={weight_decay}...")
    else:
        raise ValueError("Invalid optimizer_type!")
    return optimizer

def choose_lr_scheduler(lr_scheduler, optimizer, num_epochs):
    if lr_scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
        print("Using CosineAnnealingLR scheduler...")
    elif lr_scheduler == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        print("Using ExponentialLR scheduler...")
    elif lr_scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        print("Using ReduceLROnPlateau scheduler...")
    else:
        scheduler = None
    return scheduler

def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, loss_function, margin, threshold, patience=0, lr_scheduler=None, weight_decay=0, optimizer_type="Adam"):
    """
    Training Loop for Siamese network model.

    Args:
    - model (nn.Module): The Siamese network model to train.
    - train_loader (DataLoader): DataLoader for training data.
    - valid_loader (DataLoader): DataLoader for validation data.
    - num_epochs (int): Number of epochs to train for.
    - learning_rate (float): Learning rate for the optimizer.
    - loss_function (str): The type of loss function to be used - 'contrastive' or 'BCE'.

    Returns:
    - model (nn.Module): The trained model.
    - train_losses (list): List of training losses per epoch.
    - valid_losses (list): List of validation losses per epoch.
    """
    model = model.to(device)
    
    # Decide Loss function
    criterion = choose_loss_function(loss_function, margin)
    
    # Decide optimizer
    optimizer = choose_optimizer(model, optimizer_type, learning_rate, weight_decay)
    
    # Define learning rate scheduler
    scheduler = choose_lr_scheduler(lr_scheduler, optimizer, num_epochs)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []
    best_valid_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, num_epochs+1):
        model.train()
        running_train_loss = 0.0
        correct_train_predictions = 0  # Track correct training predictions
        total_train_samples = 0
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        
        for _, (pairs, labels) in enumerate(pbar):
            
            input1, input2 = pairs[0].to(device), pairs[1].to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            output1, output2 = model(input1, input2)
            
            # Using cosine similarity
            if loss_function == "contrastive":
                distance = F.pairwise_distance(output1, output2)
                preds = (distance < threshold).float()  # Use threshold for predictions based on distance
            else:
                similarity = F.cosine_similarity(output1, output2)
                preds = (similarity > threshold).float()  # Use threshold for predictions based on similarity

            correct_train_predictions += (preds == labels).sum().item()  # Compute correct predictions
            total_train_samples += labels.size(0)
            
            if loss_function == "BCE":
                loss = criterion(similarity, labels)
            elif loss_function == "hinge_loss":
                hinge_labels = 2*labels - 1
                loss = criterion(similarity, hinge_labels, torch.ones_like(labels))
            elif loss_function == "contrastive":
                loss = criterion(output1, output2, 1-labels)

            running_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(train_loss=loss.item())
        
        avg_train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train_predictions / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation
        model.eval()
        running_valid_loss = 0.0
        correct_valid_predictions = 0  # Track correct validation predictions
        total_valid_samples = 0
        
        for _, (pairs, labels) in enumerate(valid_loader):
            input1, input2 = pairs[0].to(device), pairs[1].to(device)
            labels = labels.to(device).float()
            
            with torch.no_grad():
                output1, output2 = model(input1, input2)
                
                if loss_function == "contrastive":
                    distance = F.pairwise_distance(output1, output2)
                    preds = (distance < threshold).float()  # Use threshold for predictions based on distance
                else:
                    similarity = F.cosine_similarity(output1, output2)
                    preds = (similarity > threshold).float()  # Use threshold for predictions based on similarity

                correct_valid_predictions += (preds == labels).sum().item()  # Compute correct predictions
                total_valid_samples += labels.size(0)
                
                if loss_function == "BCE":
                    loss = criterion(similarity, labels)
                elif loss_function == "hinge_loss":
                    hinge_labels = 2*labels - 1
                    loss = criterion(similarity, hinge_labels, torch.ones_like(labels))
                elif loss_function == "contrastive":
                    loss = criterion(output1, output2, 1-labels)

                running_valid_loss += loss.item()
        
        avg_valid_loss = running_valid_loss / len(valid_loader)
        valid_accuracy = correct_valid_predictions / total_valid_samples 
        valid_losses.append(avg_valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        # Learning Rate Scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_valid_loss)
            else:
                scheduler.step()

        # Early Stopping
        if patience > 0:
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                epochs_without_improvement = 0
                # Save best model weights
                best_model_weights = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == patience:
                    print(f"Early stopping!!, Since validation loss has not improved in the last {patience} epochs.")
                    # Load best model weights
                    model.load_state_dict(best_model_weights)
                    break

        # Print epoch details
        print(f'''{"#"*100}
Epoch: [{epoch}/{num_epochs}] | Epoch Train Loss: {avg_train_loss} | Epoch Valid Loss: {avg_valid_loss}
{"#"*100}''')
            
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies

def plot_losses(train_losses, valid_losses, train_accuracies, valid_accuracies):
    """
    Plots training and validation losses and accuracies per epoch.
    
    Args:
    - train_losses (list): Training losses per epoch.
    - valid_losses (list): Validation losses per epoch.
    - train_accuracies (list): Training accuracies per epoch.
    - valid_accuracies (list): Validation accuracies per epoch.
    """
    epochs = range(1, len(train_losses) + 1)  # Create a list of epochs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Create a figure and two axes

    # Plotting train and valid losses on the left subplot
    ax1.plot(epochs, train_losses, label='Training Loss', marker='o', color='blue')
    ax1.plot(epochs, valid_losses, label='Validation Loss', marker='o', color='red')
    ax1.set_title('Training & Validation Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting train and valid accuracies on the right subplot
    ax2.plot(epochs, train_accuracies, label='Training Accuracy', marker='o', color='blue')
    ax2.plot(epochs, valid_accuracies, label='Validation Accuracy', marker='o', color='red')
    ax2.set_title('Training & Validation Accuracies')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
def evaluate_model(model, dataloader, threshold, loss_function):
    """
    Evaluate the Siamese network model on given dataloader.

    Args:
    - model (nn.Module): The trained Siamese network model.
    - dataloader (DataLoader): DataLoader for evaluation data.
    - threshold (float): Threshold for cosine similarity to make binary predictions.

    Returns:
    - avg_accuracy (float): Average accuracy on the dataloader.
    - class_report (str): Classification report with precision, recall, etc.
    """
    model = model.to(device)
    model.eval()
    
    total_samples = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    
    for _, (pairs, labels) in enumerate(dataloader):
        input1, input2 = pairs[0].to(device), pairs[1].to(device)
        labels = labels.to(device).float()
        
        with torch.no_grad():
            output1, output2 = model(input1, input2)
            if loss_function == "contrastive":
                distance = F.pairwise_distance(output1, output2)
                preds = (distance < threshold).float()  # Use threshold for predictions based on distance
            else:
                similarity = F.cosine_similarity(output1, output2)
                preds = (similarity > threshold).float()  # Use threshold for predictions based on similarity

            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Store predictions and labels for classification report
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_accuracy = correct_predictions / total_samples
    class_report = classification_report(all_labels, all_preds, target_names=["Negative Pair", "Positive Pair"])
    
    return avg_accuracy, class_report

def display_predictions(model, dataloader, threshold, title, loss_function):
    model.eval()
    model = model.to(device)
    count = 0

    # Creating a main figure for our 8 pairs of images
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    
    for (pairs, actual_labels) in dataloader:
        if count >= 8:
            break

        input1, input2 = pairs[0].to(device), pairs[1].to(device)
        output1, output2 = model(input1, input2)
        
        if loss_function == "contrastive":
            distance = F.pairwise_distance(output1, output2).cpu().detach().numpy()
            predicted_labels = (distance < threshold).astype(float)
        else:
            similarity = F.cosine_similarity(output1, output2).cpu().detach().numpy()
            predicted_labels = (similarity > threshold).astype(float)

        for idx in range(len(predicted_labels)):
            # Setting the title with predicted and actual labels
            pair_title = f"Predicted: {'Same Person' if predicted_labels[idx] == 1.0 else 'Different Person'} | Actual: {'Same Person' if actual_labels[idx] == 1.0 else 'Different Person'}"
            
            # Denormalizing the images and ensure we're working with single images, not batches
            img1 = denormalize(pairs[0][idx].unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            img2 = denormalize(pairs[1][idx].unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)

            # Display the two images concatenated side by side
            axes[count//2, count%2].imshow(np.concatenate((img1, img2), axis=1))
            
            # Setting the title with color based on match
            title_color = 'red' if predicted_labels[idx] != actual_labels[idx] else 'black'
            axes[count//2, count%2].set_title(pair_title, size=18, x=0.47, color=title_color)

            # Removing axes
            axes[count//2, count%2].axis('off')

            count += 1
            if count >= 8:
                break
    
    # Setting the main title for the entire figure
    fig.suptitle(title, size=24, weight='bold', y=1)
    plt.tight_layout()
    plt.show()

### PART B: GANs ####
def preprocess_data_for_GAN(persons_dict):
    """
    Preprocess the images in the data directory using albumentations by applying a center crop and normalization.

    Args:
    - persons_dict (dict): Dictionary with person names as keys and the number of images as values.

    Returns:
    - X_processed (list): List of preprocessed images.
    """
    data_dir = pl.Path('./dataset/lfw/')
    transforms = A.Compose([
        A.CenterCrop(128, 128),
        # A.Resize(64, 64),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    X_processed = []

    for person, _ in persons_dict.items():
        person_dir = data_dir / person
        images = list(person_dir.glob("*.jpg"))
        for img_path in images:
            img = Image.open(img_path)
            img = np.array(img)
            img = transforms(image=img)['image']
            X_processed.append(img) 
    print(f"{len(X_processed)} images preprocessed.")
    return X_processed

def denormalize_data_for_GAN(tensor):
    """
    Denormalize a tensor.

    Args:
    - tensor (torch.Tensor): The normalized tensor.
    - mean (list): The mean values used for normalization.
    - std (list): The standard deviation values used for normalization.

    Returns:
    - torch.Tensor: The denormalized tensor.
    """
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

class GAN_Dataset(Dataset):
    """
    Dataset class for GANs.

    Attributes:
    - X (list): List of images.

    Methods:
    - __len__(): Returns the number of images in the dataset.
    - __getitem__(index): Returns the image at the given index.
    """
    def __init__(self, X):
        """
        Initializes the GAN_Dataset with given images.

        Args:
        - X (list): List of images.
        """
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

def visualize_batch_for_GAN(dataloader, num_samples=16):
    """
    Visualize a batch of images from the GAN's dataloader.

    Args:
    - dataloader (DataLoader): DataLoader object for the GAN.
    - num_samples (int): Number of samples to visualize, default is 16.
    """
    
    img_batch = next(iter(dataloader))
    
    # Denormalize the images
    img_batch = denormalize_data_for_GAN(img_batch)
    
    # Randomly sample indices for visualization
    indices = random.sample(range(len(img_batch)), num_samples)
    
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    for i, idx in enumerate(indices):
        img = np.clip(img_batch[idx].permute(1, 2, 0).numpy(), 0, 1)
        row, col = divmod(i, 4)
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    
class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 32, 4, 1, 0),  # 4x4
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # 8x8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 16x16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 32x32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 64x64
            nn.ConvTranspose2d(features_g * 2, channels_img, 4, 2, 1),  # 128x128
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # N x channels_img x 128 x 128
            nn.Conv2d(channels_img, features_d, 4, 2, 1),  # 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 32x32
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 16, 1, 4, 2, 0),  # 1x1
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)
    
def save_generated_imgs(generator, epoch, fixed_noise, num_epochs, folder_name):
    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    denormalized_images = denormalize_data_for_GAN(fake_images)    # take 64 imgs to save
    img_path = gen_imgs_dir / f"epoch_{epoch}.png"
    save_image(denormalized_images, img_path, normalize=True, nrow=8)  # 8x8 grid

    # If it's the last epoch, generate a GIF of all saved images
    if epoch == num_epochs:
        images = [Image.open(gen_imgs_dir / f"epoch_{e}.png") for e in range(1, num_epochs + 1)]
        gif_path = gen_imgs_dir / "generated_images.gif"
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0,
        )
        
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            
def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train_WGAN(generator, discriminator, train_loader, num_epochs, folder_name, learning_rate, z_dim = 100, lr_scheduler=None, disc_iterations=5, lambda_gp=10):

    # Move models to device
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    
    # Define learning rate scheduler
    scheduler_g = choose_lr_scheduler(lr_scheduler, optimizer_g, num_epochs)
    scheduler_d = choose_lr_scheduler(lr_scheduler, optimizer_d, num_epochs)

    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    if gen_imgs_dir.exists():
        shutil.rmtree(gen_imgs_dir)
    gen_imgs_dir.mkdir(exist_ok=True)  
      
    losses_g, losses_d = [], []
    real_scores, fake_scores = [], []

    for epoch in range(1, num_epochs+1):
        generator.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        running_real_score = 0.0
        running_fake_score = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        
        for real_images in pbar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            for _ in range(disc_iterations):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = generator(noise)
                # Real images
                critic_real = discriminator(real_images).reshape(-1)
                # Fake images
                critic_fake = discriminator(fake_images.detach()).reshape(-1)
                gp = gradient_penalty(discriminator, real_images, fake_images, device=device)
                loss_d = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()

            # Train Generator
            fake_images = generator(noise)
            gen_fake = discriminator(fake_images).reshape(-1)
            loss_g = -torch.mean(gen_fake)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()
            
            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()
            running_real_score += critic_real.mean().item()
            running_fake_score += critic_fake.mean().item()

            # Update tqdm
            pbar.set_postfix(gen_loss=loss_g.item(), disc_loss=loss_d.item())  
            
        # Learning Rate Scheduling
        if scheduler_g:
            if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_g.step(running_loss_g)
            else:
                scheduler_g.step()
        if scheduler_d:
            if isinstance(scheduler_d, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_d.step(running_loss_d)
            else:
                scheduler_d.step()
        
        avg_loss_g = running_loss_g / len(train_loader)
        avg_loss_d = running_loss_d / len(train_loader)
        avg_real_score = running_real_score / len(train_loader)
        avg_fake_score = running_fake_score / len(train_loader)

        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        
        fixed_noise = torch.randn(64, z_dim, 1, 1).to(device)
        save_generated_imgs(generator, epoch, fixed_noise, num_epochs, folder_name)
        
        # Print epoch details
        print(f'''{"#"*100}
Epoch: [{epoch}/{num_epochs}] | Gen Loss: {avg_loss_g:.5f} | Disc Loss: {avg_loss_d:.5f} | Real Score: {avg_real_score:.5f} | Fake Score: {avg_fake_score:.5f}
{"#"*100}''')
            
    return losses_g, losses_d, real_scores, fake_scores

def plot_GAN_loss(losses_g, losses_d, real_scores, fake_scores):
    """
    Plot the training losses of the Generator and the Discriminator, and the real vs. fake scores.
    
    Args:
    - losses_g (list): List of training losses for the Generator.
    - losses_d (list): List of training losses for the Discriminator.
    - real_scores (list): List of real scores per epoch.
    - fake_scores (list): List of fake scores per epoch.
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plotting the Generator and Discriminator loss on the left subplot
    ax1.plot(losses_g, label='Generator Loss', color='blue')
    ax1.plot(losses_d, label='Discriminator Loss', color='red')
    ax1.set_title("Generator vs. Discriminator Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plotting the real vs. fake scores on the right subplot
    ax2.plot(real_scores, label='Real Score', color='green')
    ax2.plot(fake_scores, label='Fake Score', color='orange')
    ax2.set_title("Real vs. Fake Scores")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    
def visualize_GAN_results(num_epochs, every_n_epochs, folder_name):
    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    images_to_display = []

    for epoch in range(1, num_epochs + 1):
        if epoch % every_n_epochs == 0:
            img_path = gen_imgs_dir / f"epoch_{epoch}.png"
            if img_path.exists():
                images_to_display.append(plt.imread(img_path))

    # Plotting the images
    num_images = len(images_to_display)
    fig, axes = plt.subplots(num_images // 2, 2, figsize=(10, num_images * 2))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images_to_display):
            ax.imshow(images_to_display[idx])
            ax.axis("off")
            ax.set_title(f"Epoch {every_n_epochs * (idx+1)}", fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
class cGAN_Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, condition_dim):
        super(cGAN_Generator, self).__init__()
        self.condition_dim = condition_dim
        self.net = nn.Sequential(
            self._block(channels_noise + condition_dim, features_g * 32, 4, 1, 0),  # img: 4x4
            self._block(features_g * 32, features_g * 16, 4, 2, 1),  # img: 8x8
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 16x16
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 32x32
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 64x64
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),  # img: 128x128
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, noise, condition):
        x = torch.cat((noise, condition), 1)
        return self.net(x)

class cGAN_Discriminator(nn.Module):
    def __init__(self, channels_img, features_d, condition_dim):
        super(cGAN_Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channels_img + condition_dim, features_d, kernel_size=4, stride=2, padding=1),  # img: 64x64
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # img: 32x32
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # img: 16x16
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # img: 8x8
            self._block(features_d * 8, features_d * 16, 4, 2, 1),  # img: 4x4
            nn.Conv2d(features_d * 16, 1, kernel_size=4, stride=2, padding=0),  # img: 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, image, condition):
        condition = condition.view(condition.size(0), condition.size(1), 1, 1)
        condition = condition.repeat(1, 1, image.size(2), image.size(3))
        x = torch.cat((image, condition), 1)
        return self.disc(x)

def cGAN_gradient_penalty(critic, real, fake, condition, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, condition).reshape(-1)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def cGAN_save_generated_imgs(generator, real_images, condition, epoch, fixed_noise, num_epochs, folder_name):
    fixed_noise = fixed_noise[:64]
    condition = condition[:64]
    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    generator.eval()  # Set generator to evaluation mode
    condition = condition[:64]    
    # Save Real Images
    denormalized_real_images = denormalize_data_for_GAN(real_images.cpu())
    real_img_path = gen_imgs_dir / f"real_{epoch}.png"
    save_image(denormalized_real_images, real_img_path, normalize=True, nrow=8)  # 8x8 grid

    # Generate and Save Fake Images
    with torch.no_grad():
        fake_images = generator(fixed_noise, condition).detach().cpu()
    denormalized_fake_images = denormalize_data_for_GAN(fake_images)
    fake_img_path = gen_imgs_dir / f"fake_{epoch}.png"
    save_image(denormalized_fake_images, fake_img_path, normalize=True, nrow=8)  # 8x8 grid

    # If it's the last epoch, generate separate GIFs for real and fake images
    if epoch == num_epochs:
        # Generate GIF for real images
        real_images = [Image.open(gen_imgs_dir / f"real_{e}.png") for e in range(1, num_epochs + 1)]
        real_gif_path = gen_imgs_dir / "real_images.gif"
        real_images[0].save(
            real_gif_path,
            save_all=True,
            append_images=real_images[1:],
            duration=100,
            loop=0,
        )

        # Generate GIF for fake images
        fake_images = [Image.open(gen_imgs_dir / f"fake_{e}.png") for e in range(1, num_epochs + 1)]
        fake_gif_path = gen_imgs_dir / "fake_images.gif"
        fake_images[0].save(
            fake_gif_path,
            save_all=True,
            append_images=fake_images[1:],
            duration=100,
            loop=0,
        )

def train_cGAN(generator, discriminator, siamese_net, train_loader, num_epochs, folder_name, learning_rate, margin, z_dim = 100, lr_scheduler=None, disc_iterations=5, lambda_gp=10, lambda_siamese=0.1):
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    siamese_net = siamese_net.to(device)

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.9))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.9))

    # Define learning rate scheduler
    scheduler_g = choose_lr_scheduler(lr_scheduler, optimizer_g, num_epochs)
    scheduler_d = choose_lr_scheduler(lr_scheduler, optimizer_d, num_epochs)

    # Contrastive loss for Siamese network
    contrastive_criterion = ContrastiveLoss(margin=margin)
    
    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    if gen_imgs_dir.exists():
        shutil.rmtree(gen_imgs_dir)
    gen_imgs_dir.mkdir(exist_ok=True) 

    losses_g, losses_d = [], []
    real_scores, fake_scores = [], []
    
    for epoch in range(1, num_epochs+1):
        generator.train()
        discriminator.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        running_real_score = 0.0
        running_fake_score = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), leave=False)
        for real_images in pbar:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Get embeddings of real images using Siamese network
            real_embeddings = siamese_net.forward_once(real_images).unsqueeze(2).unsqueeze(3)

            # Train Discriminator
            for _ in range(disc_iterations):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake_images = generator(noise, real_embeddings)

                # Discriminator forward
                critic_real = discriminator(real_images, real_embeddings).reshape(-1)
                critic_fake = discriminator(fake_images.detach(), real_embeddings).reshape(-1)

                gp = cGAN_gradient_penalty(discriminator, real_images, fake_images, real_embeddings, device=device)
                loss_d = -(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp * gp
                optimizer_d.zero_grad()
                loss_d.backward(retain_graph=True)
                optimizer_d.step()

            # Train Generator
            fake_images = generator(noise, real_embeddings)
            gen_fake = discriminator(fake_images, real_embeddings).reshape(-1)

            # Siamese loss: Ensuring that the embeddings of real and fake images are similar but not identical
            fake_embeddings = siamese_net.forward_once(fake_images).unsqueeze(2).unsqueeze(3)
            siamese_loss = contrastive_criterion(real_embeddings, fake_embeddings, torch.ones(batch_size,).to(device))

            loss_g = -torch.mean(gen_fake) + lambda_siamese * siamese_loss
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()
            running_real_score += critic_real.mean().item()
            running_fake_score += critic_fake.mean().item()
            pbar.set_postfix(gen_loss=loss_g.item(), disc_loss=loss_d.item())
                  
        # Learning Rate Scheduling
        if scheduler_g:
            if isinstance(scheduler_g, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_g.step(running_loss_g)
            else:
                scheduler_g.step()
        if scheduler_d:
            if isinstance(scheduler_d, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_d.step(running_loss_d)
            else:
                scheduler_d.step()
        
        avg_loss_g = running_loss_g / len(train_loader)
        avg_loss_d = running_loss_d / len(train_loader)
        avg_real_score = running_real_score / len(train_loader)
        avg_fake_score = running_fake_score / len(train_loader)

        losses_g.append(avg_loss_g)
        losses_d.append(avg_loss_d)
        real_scores.append(avg_real_score)
        fake_scores.append(avg_fake_score)
        
        fixed_noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        
        cGAN_save_generated_imgs(generator, real_images[:64], real_embeddings[:64], epoch, fixed_noise, num_epochs, folder_name)
        
        # Print epoch details
        print(f'''{"#"*100}
Epoch: [{epoch}/{num_epochs}] | Gen Loss: {avg_loss_g:.5f} | Disc Loss: {avg_loss_d:.5f} | Real Score: {avg_real_score:.5f} | Fake Score: {avg_fake_score:.5f}
{"#"*100}''')
            
    return losses_g, losses_d, real_scores, fake_scores

def visualize_cGAN_results(num_epochs, every_n_epochs, folder_name):
    gen_imgs_dir = pl.Path(f'./generated_images/{folder_name}')
    
    # Lists to store paths of real and fake images for each epoch
    real_images_paths = [gen_imgs_dir / f"real_{e}.png" for e in range(1, num_epochs + 1) if (e % every_n_epochs == 0)]
    fake_images_paths = [gen_imgs_dir / f"fake_{e}.png" for e in range(1, num_epochs + 1) if (e % every_n_epochs == 0)]
    
    # Lists to store the actual image data
    real_images_to_display = [plt.imread(path) for path in real_images_paths]
    fake_images_to_display = [plt.imread(path) for path in fake_images_paths]

    num_images = len(real_images_to_display)
    
    fig, axes = plt.subplots(num_images, 2, figsize=(14, num_images * 6))
    
    for i in range(num_images):
        # Displaying Fake Images on the left
        axes[i, 0].imshow(fake_images_to_display[i])
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Fake_{every_n_epochs * (i+1)}", fontsize=20)
        
        # Displaying Real Images on the right
        axes[i, 1].imshow(real_images_to_display[i])
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Real_{every_n_epochs * (i+1)}", fontsize=20)
    
    plt.tight_layout()
    plt.show()
        
def main(**kwargs):
    hyperparameters = {
    'batch_size': 64,
    'random_seed': 42,
    'epochs': 20,
    'learning_rate': 0.01,
    'base_model': 'resnet18',
    'margin': 1.0,
    'threshold': 0.5,
    'max_positive_combinations': 10,
    'loss_function': 'contrastive',
    'patience': 0,
    'lr_scheduler': None, # 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau'
    'optimizer_type': 'Adam', # 'Adam', 'Adagrad', 'RMSprop'
    'weight_decay': 0,
    'apply_augmentation': True,
    'unfreeze_last_n': 0,
    'num_augmentations': 1,
    'print_tables': True,
    'plot_losses': True,
    'display_predictions': True,
    'visualize_pairs': True,
    'save_best': False
    }
    hyperparameters.update(kwargs)
    print(f'{"#"*100}\nMetric Learning & Generative AI\n{"#"*100}')
    print(f"Hyperparameters: {hyperparameters}")

    # Override default hyperparameterss with provided arguments
    for key, value in kwargs.items():
        if key in hyperparameters:
            hyperparameters[key] = value

    data_dir = pl.Path('./dataset/lfw/')     # Setting path to the data directory
    persons_dirs = [d for d in data_dir.iterdir() if d.is_dir()] # List all subdirectories (each directory corresponds to a person)
    all_persons_dict = {person_dir.name: len(list(person_dir.glob("*.jpg"))) for person_dir in persons_dirs} # Create a dictionary of all persons and the number of images they have
    all_persons = list(all_persons_dict.keys())             # List of all persons 
    persons_with_mul_imgs_dict = {person: num_images for person, num_images in all_persons_dict.items() if num_images > 1} 

    if hyperparameters['train_Siamese']:
        train_persons_dict, valid_persons_dict, test_persons_dict = split_data(all_persons_dict)
        
        if hyperparameters['apply_augmentation']:
            print("Generating augmented images...")
            generate_positive_pairs(train_persons_dict, num_augmentations=hyperparameters['num_augmentations'])
            
        X_train_pairs, Y_train_pairs, positive_pairs_count_train, negative_pairs_count_train = generate_pairs(train_persons_dict, max_positive_combinations=hyperparameters['max_positive_combinations'], apply_augmentation=hyperparameters['apply_augmentation'], num_augmentations=hyperparameters['num_augmentations'])
        X_valid_pairs, Y_valid_pairs, positive_pairs_count_valid, negative_pairs_count_valid = generate_pairs(valid_persons_dict, max_positive_combinations=1, apply_augmentation=False, num_augmentations=0)
        X_test_pairs, Y_test_pairs, positive_pairs_count_test, negative_pairs_count_test = generate_pairs(test_persons_dict, max_positive_combinations=1, apply_augmentation=False, num_augmentations=0)

        if hyperparameters['print_tables']:
            print(f"Toal number of persons: {len(all_persons)}")
            print(f"Number of persons with more than one image: {len(persons_with_mul_imgs_dict)}")
            table = PrettyTable()
            table.field_names = ["Dataset Split", "Number of Persons", "Number of Positive Pairs", "Number of Negative Pairs"]
            table.add_row(["Training Set", len(train_persons_dict), positive_pairs_count_train, negative_pairs_count_train])
            table.add_row(["Validation Set", len(valid_persons_dict), positive_pairs_count_valid, negative_pairs_count_valid])
            table.add_row(["Test Set", len(test_persons_dict), positive_pairs_count_test, negative_pairs_count_test])
            print(table)
            
        print("Creating training, validation, and test datasets...")
        X_train, Y_train = dict_to_tensors(train_persons_dict, max_positive_combinations=hyperparameters['max_positive_combinations'], apply_augmentation=hyperparameters['apply_augmentation'], num_augmentations=hyperparameters['num_augmentations'])
        X_valid, Y_valid = dict_to_tensors(valid_persons_dict, max_positive_combinations=1, apply_augmentation=False, num_augmentations=0)
        X_test, Y_test = dict_to_tensors(test_persons_dict, max_positive_combinations=1, apply_augmentation=False, num_augmentations=0)
        
        train_dataset = SiameseDataset(X_train, Y_train)
        valid_dataset = SiameseDataset(X_valid, Y_valid)
        test_dataset = SiameseDataset(X_test, Y_test)
        train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=16, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False, num_workers=16, pin_memory=True)
        
        if hyperparameters['visualize_pairs']:
            visualize_pairs(X_train_pairs, Y_train_pairs)  
        
        print(f"Training Siamese Network with '{hyperparameters['base_model']}' base...")     
        # Initialize model
        model = SiameseNetwork(base_model=hyperparameters['base_model'], unfreeze_last_n=hyperparameters['unfreeze_last_n'])
        
        # Train model
        trained_model, train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
            model, 
            train_loader, 
            valid_loader, 
            hyperparameters['epochs'], 
            hyperparameters['learning_rate'],
            hyperparameters['loss_function'],
            hyperparameters['margin'],
            hyperparameters['threshold'],
            patience=hyperparameters['patience'],
            lr_scheduler=hyperparameters['lr_scheduler'],
            weight_decay=hyperparameters['weight_decay'],
            optimizer_type=hyperparameters['optimizer_type']
        )
        test_accuracy, class_report = evaluate_model(trained_model, test_loader, threshold=hyperparameters['threshold'], loss_function = hyperparameters['loss_function'])
        accuracy_table = PrettyTable()
        accuracy_table.title = f"Model Performance with '{hyperparameters['base_model']}' base"
        accuracy_table.field_names = ["Split", "Accuracy (%)"]
        accuracy_table.add_row(["Training", f"{np.mean(train_accuracies)*100:.2f}"])
        accuracy_table.add_row(["Validation", f"{np.mean(valid_accuracies)*100:.2f}"])
        accuracy_table.add_row(["Test", f"{test_accuracy*100:.2f}"])
        print(accuracy_table)
        print(f"Classification Report:\n{class_report}")
        
        if hyperparameters['save_best']:
            save_dir = pl.Path('./best_models/')
            save_dir.mkdir(exist_ok=True)
            best_model_name = f"Best_Model_val_acc_{np.mean(valid_accuracies)*100:.2f}_test_acc_{test_accuracy*100:.2f}.pth"
            torch.save(trained_model.state_dict(), save_dir / best_model_name)
            print(f"Saving best model to '{save_dir / best_model_name}'...")
        
        # Optionally plot losses
        if hyperparameters['plot_losses']:
            plot_losses(train_losses, valid_losses, train_accuracies, valid_accuracies)
        
        # Optionally display predictions
        if hyperparameters['display_predictions']:
            display_predictions(trained_model, valid_loader, threshold=hyperparameters['threshold'], title=f"Predictions on Validation Set with '{hyperparameters['base_model']}' base", loss_function = hyperparameters['loss_function'])
            display_predictions(trained_model, test_loader, threshold=hyperparameters['threshold'], title=f"Predictions on Test Set with '{hyperparameters['base_model']}' base", loss_function = hyperparameters['loss_function'])
            
    if hyperparameters['train_WGAN']:
        X_processed = preprocess_data_for_GAN(all_persons_dict)
        gan_dataset = GAN_Dataset(X_processed)
        gan_loader = DataLoader(gan_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        
        print("Training WGAN Model...")
        generator = Generator(channels_noise=128, channels_img=3, features_g=16).to(device)
        discriminator = Discriminator(channels_img=3, features_d=16).to(device)
        initialize_weights(generator)
        initialize_weights(discriminator)
        losses_g, losses_d, real_scores, fake_scores = train_WGAN(generator, discriminator, gan_loader, num_epochs=hyperparameters['epochs'], folder_name='WGAN', learning_rate=1e-4, z_dim=128, lr_scheduler=None, disc_iterations=5, lambda_gp=10)
        print("Saving generated images to './generated_images/WGAN/' folder...")
        
        if hyperparameters['visualize_GAN_results']:
            visualize_GAN_results(num_epochs=hyperparameters['epochs'], every_n_epochs=10, folder_name='WGAN')
            
        if hyperparameters['plot_GAN_losses']:
            plot_GAN_loss(losses_g, losses_d, real_scores, fake_scores)
    
    if hyperparameters['train_cGAN']:
        X_processed = preprocess_data_for_GAN(all_persons_dict)
        gan_dataset = GAN_Dataset(X_processed)
        gan_loader = DataLoader(gan_dataset, batch_size=hyperparameters['batch_size'], shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
        
        print("Training cGAN Model...")
        cgan_generator = cGAN_Generator(channels_noise=128, channels_img=3, features_g=16, condition_dim=512).to(device)
        cgan_discriminator = cGAN_Discriminator(channels_img=3, features_d=16, condition_dim=512).to(device)
        initialize_weights(cgan_generator)
        initialize_weights(cgan_discriminator)
        siamese_net = SiameseNetwork(base_model='resnet18', unfreeze_last_n=-1).to(device)
        siamese_net_weights_path = './best_models/Best_Model_val_acc_78.30_test_acc_83.63.pth'
        print(f"Loading Siamese Network weights from '{siamese_net_weights_path}'...")
        siamese_net.load_state_dict(torch.load(siamese_net_weights_path))
        siamese_net = siamese_net.to(device)
        siamese_net.eval() 
        losses_g, losses_d, real_scores, fake_scores = train_cGAN(cgan_generator, cgan_discriminator, siamese_net, gan_loader, num_epochs=hyperparameters['epochs'], folder_name='cGAN', learning_rate=1e-4, margin=1.0, z_dim=128, lr_scheduler=None, disc_iterations=2, lambda_gp=10, lambda_siamese=0.01)
        print("Saving generated images to './generated_images/cGAN/' folder...")
        
        if hyperparameters['visualize_GAN_results']:
            visualize_cGAN_results(num_epochs=hyperparameters['epochs'], every_n_epochs=10, folder_name='cGAN')
            
        if hyperparameters['plot_GAN_losses']:
            plot_GAN_loss(losses_g, losses_d, real_scores, fake_scores)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Siamese Network')

    # Training arguments
    parser.add_argument('--train_Siamese', type=bool, default=False, help='Flag to train Siamese Network.')
    parser.add_argument('--train_WGAN', type=bool, default=False, help='Flag to train WGAN.')
    parser.add_argument('--train_cGAN', type=bool, default=False, help='Flag to train cGAN.')
    
    # Model Hyperparameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--base_model', type=str, default='resnet18', help='Base model for Siamese Network.')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for contrastive loss.')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for similarity prediction.')
    parser.add_argument('--max_positive_combinations', type=int, default=10, help='Maximum number of positive combinations per person.')
    parser.add_argument('--loss_function', type=str, default='contrastive', choices=['BCE', 'hinge_loss','contrastive'], help='Loss function to use for training.')
    parser.add_argument('--patience', type=int, default=4, help='Patience for early stopping.')
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'CosineAnnealingLR', 'ExponentialLR', 'ReduceLROnPlateau'], help='Learning rate scheduler.')
    parser.add_argument('--optimizer_type', type=str, default='Adam', choices=['Adam', 'Adagrad', 'RMSprop'], help='Optimizer type.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for the optimizer.')
    parser.add_argument('--apply_augmentation', type=bool, default=True, help='Whether to apply data augmentation or not.')
    parser.add_argument('--print_tables', type=bool, default=False, help='Flag to print tables.')
    parser.add_argument('--num_augmentations', type=int, default=4, help='Number of augmented images to generate per image.')
    parser.add_argument('--unfreeze_last_n', type=int, default=4, help='Number of layers to unfreeze from the end.')
    parser.add_argument('--save_best', type=bool, default=True, help='Flag to save best model.')
    
    # Visualization arguments
    parser.add_argument('--plot_losses', type=bool, default=False, help='Flag to plot training and validation losses.')
    parser.add_argument('--display_predictions', type=bool, default=False, help='Flag to display predictions on validation and test sets.')
    parser.add_argument('--visualize_pairs', type=bool, default=False, help='Flag to visualize pairs of images.')
    parser.add_argument('--visualize_GAN_results', type=bool, default=False, help='Flag to visualize GAN results.')
    parser.add_argument('--plot_GAN_losses', type=bool, default=False, help='Flag to plot GAN losses.')
    args = parser.parse_args()
    hyperparams = {key: value for key, value in vars(args).items()}
    main(**hyperparams)