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

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

def visualize_dataset(persons_dict):
    """
    Visualizes a grid of persons' images.
    
    Args:
    - persons_dict (dict): Dictionary with person names as keys and the number of images as values.
    """
    # Display in an 8x8 grid
    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    
    grid_position = 0  # Counter for the position in the 8x8 grid
    
    for person, count in persons_dict.items():
        if grid_position >= 64:  # Break after 64 cells
            break
        person_dir = data_dir / person
        images = list(person_dir.glob("*.jpg"))
        for img_path in images:
            if grid_position >= 64:  # Break after 64 cells
                break
            image = Image.open(img_path)
            image = image.resize((64, 64))
            row, col = divmod(grid_position, 8)
            axes[row, col].imshow(image)
            axes[row, col].axis('off')
            axes[row, col].set_title(f"{person}")
            grid_position += 1

    plt.tight_layout()
    plt.show()
    
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Siamese Network')

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
    args = parser.parse_args()
    hyperparams = {key: value for key, value in vars(args).items()}
    main(**hyperparams)