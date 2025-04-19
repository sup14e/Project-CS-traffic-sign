# ###########################
# # Import Python Packages
# ###########################
# import os
# from datetime import datetime

# import torch
# from torch import nn
# from torch.utils.data import DataLoader, random_split
# from torch.utils.tensorboard import SummaryWriter
# import torchvision.transforms.v2 as transforms_v2
# from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
# import torch.nn.functional as F

# from dl_utils import train_one_epoch, test, plot_predictions
# from model import Model1, Model2, Model3, Model4, Model5
# from dataset import ObjectDataset, YOLODataset

# import pandas as pd
# import yaml

# ####################
# # Hyperparameters
# ####################
# learning_rate = 3e-4
# batch_size = 32         
# epochs = 5              


# ####################
# # Dataset
# ####################
# DATA_DIR = "new_data"
# IMG_DIR = DATA_DIR
# # LABELS_FILE = os.path.join(DATA_DIR, "annotations.csv")
# LABELS_DIR = os.path.join(DATA_DIR, "labels")  # Directory containing YOLO format label files
# YAML_PATH = os.path.join(DATA_DIR, "Custom_data.yaml")

# with open(YAML_PATH, 'r') as f:
#     data_yaml = yaml.safe_load(f)
#     class_names = data_yaml['names']


# # Load CSV labels into a dictionary
# # df = pd.read_csv(LABELS_FILE, sep=',')

# # labels = {}
# # for _, row in df.iterrows():
# #     labels[row['filename']] = {
# #         "bbox": [
# #             row['x1'], row['y1'], 
# #             row['x2'] - row['x1'],  # width
# #             row['y2'] - row['y1'],  # height
# #         ],
# #         "class": row['class']
# #     }

# # Define Augmentations for Training
# train_transforms = transforms_v2.Compose([
#     transforms_v2.Resize((224, 224)),
#     transforms_v2.RandomHorizontalFlip(p=0.5),
#     transforms_v2.RandomRotation(degrees=15),
#     transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms_v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
#     transforms_v2.ToImage(),
#     transforms_v2.ToDtype(torch.float32, scale=True),
#     transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Define Transformations for Validation & Test (No Augmentation, but with Bounding Box Support)
# test_transforms = transforms_v2.Compose([
#     transforms_v2.Resize((224, 224)),
#     transforms_v2.ToImage(),
#     transforms_v2.ToDtype(torch.float32, scale=True),
#     transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Load dataset
# # full_dataset = ObjectDataset(labels, DATA_DIR)
# # Load YOLO dataset
# full_dataset = YOLODataset(
#     img_dir=IMG_DIR,
#     label_dir=LABELS_DIR,
#     img_size=224,
#     transform=None,  # We'll apply transforms after splitting
#     num_classes=len(class_names)
# )

# # Split into train, valid, and test
# torch.manual_seed(42)
# n_valid_samples = int(0.1 * len(full_dataset))
# n_test_samples = int(0.1 * len(full_dataset))
# n_train_samples = len(full_dataset) - n_valid_samples - n_test_samples
# print("valid: " , n_valid_samples)
# print("test: " , n_test_samples)
# print("train: " , n_train_samples)

# train_ds, valid_ds, test_ds = random_split(full_dataset, [n_train_samples, n_valid_samples, n_test_samples])

# # Apply transformations
# train_ds.dataset.transform = train_transforms
# valid_ds.dataset.transform = test_transforms
# test_ds.dataset.transform = test_transforms

# # Define DataLoaders
# train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
# test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# print("works successfully :D")

# ####################
# # Model
# ####################
# device = "cuda"
# print(f"Using {device} device")

# # TODO: Create a model
# model = Model1().to(device) # YOUR CODE HERE
# # print(model)


# ####################
# # Model Training
# ####################
# writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# # TODO: Define loss functions
# class_loss_fn = nn.CrossEntropyLoss() # YOUR CODE HERE
# bbox_loss_fn = nn.MSELoss() # YOUR CODE HERE

# # TODO: Define optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # YOUR CODE HERE

# # Training loop
# best_vloss = float('inf')
# for epoch in range(epochs):
#     print(f"Epoch {epoch+1} / {epochs}")

#     train_one_epoch(
#         dataloader=train_dl, 
#         model=model, 
#         class_loss_fn=class_loss_fn, 
#         bbox_loss_fn=bbox_loss_fn, 
#         optimizer=optimizer, 
#         epoch=epoch, 
#         device=device, 
#         writer=writer,
#         log_step_interval=1,
#     )

#     # Compute train & validation loss
#     train_loss, train_bbox_loss, train_y_preds, train_y_trues, train_bbox_preds, train_bbox_trues = test(
#         train_dl, model, class_loss_fn, bbox_loss_fn, device
#     )
#     val_loss, val_bbox_loss, val_y_preds, val_y_trues, val_bbox_preds, val_bbox_trues = test(
#         valid_dl, model, class_loss_fn, bbox_loss_fn, device
#     )

#     # Compute classification metrics
#     train_accuracy = multiclass_accuracy(train_y_preds, train_y_trues).item()
#     train_f1 = multiclass_f1_score(train_y_preds, train_y_trues, average="macro", num_classes=9).item()
#     val_accuracy = multiclass_accuracy(val_y_preds, val_y_trues).item()
#     val_f1 = multiclass_f1_score(val_y_preds, val_y_trues, average="macro", num_classes=9).item()

#     # Compute bounding box MSE
#     train_bbox_mse = F.mse_loss(train_bbox_preds, train_bbox_trues).item()
#     val_bbox_mse = F.mse_loss(val_bbox_preds, val_bbox_trues).item()

#     # Log training performance
#     writer.add_scalars('Train vs. Valid/loss', 
#         {'train': train_loss, 'valid': val_loss}, 
#         epoch)
#     writer.add_scalars('Train vs. Valid/bbox_mse', 
#         {'train': train_bbox_mse, 'valid': val_bbox_mse}, 
#         epoch)
#     writer.add_scalars('Train vs. Valid/acc', 
#         {'train': train_accuracy, 'valid': val_accuracy}, 
#         epoch)
#     writer.add_scalars('Train vs. Valid/f1', 
#         {'train': train_f1, 'valid': val_f1}, 
#         epoch)

#     # Save the best model
#     if val_loss < best_vloss:
#         best_vloss = val_loss
#         torch.save(model.state_dict(), 'model_best_vloss.pth')
#         print('Saved best model to model_best_vloss.pth')

# print("Training Complete!")


# ###########################
# # Evaluate on the Test Set
# ###########################
# # TODO: Load the best model
# model = Model1().to(device) # YOUR CODE HERE
# model.load_state_dict(torch.load("model_best_vloss.pth"))

# # Evaluate on the test set
# test_loss, test_bbox_loss, test_y_preds, test_y_trues, test_bbox_preds, test_bbox_trues = test(
#     test_dl, model, class_loss_fn, bbox_loss_fn, device
# )

# # Compute test classification metrics
# test_accuracy = multiclass_accuracy(test_y_preds, test_y_trues).item()
# test_f1 = multiclass_f1_score(test_y_preds, test_y_trues, average="macro", num_classes=9).item()

# # Compute bounding box MSE
# test_bbox_mse = F.mse_loss(test_bbox_preds, test_bbox_trues).item()

# print(f"\nTest Results:")
# print(f"Classification Loss: {test_loss:.4f}")
# print(f"Bounding Box MSE: {test_bbox_mse:.4f}")
# print(f"Accuracy: {test_accuracy:.2f}%")
# print(f"F1 Score: {test_f1:.2f}")

# # Get a batch from test set
# test_images, test_labels, test_bboxes = next(iter(test_dl))

# # TODO: Make predictions
# test_preds, test_bboxes_pred = model(test_images.to(device)) # YOUR CODE HERE
# test_preds = test_preds.argmax(1)


# # Plot predictions
# plot_predictions(
#     test_images, test_labels, test_bboxes, full_dataset.idx_to_class,
#     test_preds.cpu(), test_bboxes_pred.cpu(), 
#     num_samples=8, save_path="predictions.jpg",
# )

###########################
# Import Python Packages
###########################
import os
from datetime import datetime
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.v2 as transforms_v2
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
import torch.nn.functional as F

from dl_utils import train_one_epoch, test, plot_predictions
from model import Model1, Model2, Model3, Model4, Model5
# Import your YOLO dataset
# from dataset import YOLODataset

import pandas as pd
from ultralytics import YOLODataset
from torch.utils.data import DataLoader


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    ####################
    # Hyperparameters
    ####################
    learning_rate = 1e-5
    batch_size = 32         
    epochs = 5              


    ####################
    # Dataset
    ####################
    DATA_DIR = "new_data"
    IMG_DIR = DATA_DIR
    LABELS_DIR = os.path.join(DATA_DIR, "labels")  # Directory containing YOLO format label files
    YAML_PATH = os.path.join(DATA_DIR, "Custom_data.yaml")  # Path to your YAML file with class names

    # Load class names from YAML file
    with open(YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
        class_names = data_yaml['names']
        
    # Create class index mapping
    num_classes = len(class_names)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    idx_to_class = {i: cls_name for i, cls_name in enumerate(class_names)}

    print(f"Loaded {num_classes} classes from YAML file")

    # Define Augmentations for ing
    train_transforms = transforms_v2.Compose([
        transforms_v2.Resize((224, 224)),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.RandomRotation(degrees=15),
        transforms_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms_v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define Transformations for Validation & Test (No Augmentation, but with Bounding Box Support)
    test_transforms = transforms_v2.Compose([
        transforms_v2.Resize((224, 224)),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load YOLO dataset
    
    full_dataset = YOLODataset(
        data_dir=DATA_DIR,
        img_size=224,
        # transform=None,  # We'll apply transforms after splitting
        num_classes=num_classes
    )

    # Split into train, valid, and test
    # torch.manual_seed(42)
    # n_valid_samples = int(0.1 * len(full_dataset))
    # n_test_samples = int(0.1 * len(full_dataset))
    # n_train_samples = len(full_dataset) - n_valid_samples - n_test_samples
    # print("valid: ", n_valid_samples)
    # print("test: ", n_test_samples)
    # print("train: ", n_train_samples)

    # train_ds, valid_ds, test_ds = random_split(full_dataset, [n_train_samples, n_valid_samples, n_test_samples])

    # # Apply transformations
    # train_ds.dataset.transform = train_transforms
    # valid_ds.dataset.transform = test_transforms
    # test_ds.dataset.transform = test_transforms

    # # Define DataLoaders
    # train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    # test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Create dataset
    train_dataset = YOLODataset(
        data_dir="new_data",
        split="train",
        img_size=640,
        augment=True
    )

    valid_dataset = YOLODataset(
        data_dir="new_data",
        split="valid",
        img_size=640,
        augment=True
    )

    test_dataset = YOLODataset(
        data_dir="new_data",
        split="test",
        img_size=640,
        augment=True
    )

    # Create dataloader with custom collate function
    

    # Create dataset
    dataset = YOLODataset(
        img_path="path/to/images",
        data={"path": "path/to/data.yaml"}
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=YOLODataset.collate_fn
    # )

    # valid_loader = DataLoader(
    #     valid_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=YOLODataset.collate_fn
    # )

    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=YOLODataset.collate_fn
    # )

    print("Data loading complete successfully!")

    ####################
    # Model
    ####################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Create a model - passing the number of classes from your dataset
    model = Model1(num_classes=num_classes).to(device)
    # print(model)


    ####################
    # Model Training
    ####################
    writer = SummaryWriter(f'./runs/trainer_{model._get_name()}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    # Define loss functions
    class_loss_fn = nn.CrossEntropyLoss()
    bbox_loss_fn = nn.MSELoss()

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop

    best_vloss = float('inf')
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} / {epochs}")

        train_one_epoch(
            dataloader=dataloader, 
            model=model, 
            class_loss_fn=class_loss_fn, 
            bbox_loss_fn=bbox_loss_fn, 
            optimizer=optimizer, 
            epoch=epoch, 
            device=device, 
            writer=writer,
            log_step_interval=1,
        )

        # Compute train & validation loss
        train_loss, train_bbox_loss, train_y_preds, train_y_trues, train_bbox_preds, train_bbox_trues = test(
            dataloader, model, class_loss_fn, bbox_loss_fn, device
        )
        val_loss, val_bbox_loss, val_y_preds, val_y_trues, val_bbox_preds, val_bbox_trues = test(
            dataloader, model, class_loss_fn, bbox_loss_fn, device
        )

        # Compute classification metrics
        train_accuracy = multiclass_accuracy(train_y_preds, train_y_trues).item()
        train_f1 = multiclass_f1_score(train_y_preds, train_y_trues, average="macro", num_classes=num_classes).item()
        val_accuracy = multiclass_accuracy(val_y_preds, val_y_trues).item()
        val_f1 = multiclass_f1_score(val_y_preds, val_y_trues, average="macro", num_classes=num_classes).item()

        # Compute bounding box MSE
        train_bbox_mse = F.mse_loss(train_bbox_preds, train_bbox_trues).item() 
        val_bbox_mse = F.mse_loss(val_bbox_preds, val_bbox_trues).item()

        # Log training performance
        writer.add_scalars('Train vs. Valid/loss', 
            {'train': train_loss, 'valid': val_loss}, 
            epoch)
        writer.add_scalars('Train vs. Valid/bbox_mse', 
            {'train': train_bbox_mse, 'valid': val_bbox_mse}, 
            epoch)
        writer.add_scalars('Train vs. Valid/acc', 
            {'train': train_accuracy, 'valid': val_accuracy}, 
            epoch)
        writer.add_scalars('Train vs. Valid/f1', 
            {'train': train_f1, 'valid': val_f1}, 
            epoch)

        # Save the best model
        if val_loss < best_vloss:
            best_vloss = val_loss
            torch.save(model.state_dict(), 'model_best_vloss.pth')
            print('Saved best model to model_best_vloss.pth')

    print("Training Complete!")


    ###########################
    # Evaluate on the Test Set
    ###########################
    # Load the best model
    model = Model1(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("model_best_vloss.pth"))

    # Evaluate on the test set
    test_loss, test_bbox_loss, test_y_preds, test_y_trues, test_bbox_preds, test_bbox_trues = test(
        dataloader, model, class_loss_fn, bbox_loss_fn, device
    )

    # Compute test classification metrics
    test_accuracy = multiclass_accuracy(test_y_preds, test_y_trues).item()
    test_f1 = multiclass_f1_score(test_y_preds, test_y_trues, average="macro", num_classes=num_classes).item()

    # Compute bounding box MSE
    test_bbox_mse = F.mse_loss(test_bbox_preds, test_bbox_trues).item()

    print(f"\nTest Results:")
    print(f"Classification Loss: {test_loss:.4f}")
    print(f"Bounding Box MSE: {test_bbox_mse:.4f}")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"F1 Score: {test_f1:.2f}")

    # Get a batch from test set
    for batch in dataloader:
        test_images, test_labels, test_bboxes = batch
        break

    # Make predictions
    test_preds, test_bboxes_pred = model(test_images.to(device))
    test_preds = test_preds.argmax(1)

    # Plot predictions
    plot_predictions(
        test_images, test_labels, test_bboxes, idx_to_class,
        test_preds.cpu(), test_bboxes_pred.cpu(), 
        num_samples=8, save_path="predictions.jpg",
    )