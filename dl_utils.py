import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def train_one_epoch(dataloader, model, class_loss_fn, bbox_loss_fn, optimizer, epoch, device, writer, log_step_interval=10):
    model.train()
    running_loss = 0.0
    running_class_loss = 0.0
    running_bbox_loss = 0.0
    
    for i, (images, targets, bboxes) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        bboxes = bboxes.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        class_preds, bbox_preds = model(images)
        
        # Calculate losses
        class_loss = class_loss_fn(class_preds, targets)
        bbox_loss = bbox_loss_fn(bbox_preds, bboxes)
        
        # Combined loss - bbox loss weighted slightly higher for accurate localization
        loss = class_loss + 1.5 * bbox_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update running losses
        running_loss += loss.item()
        running_class_loss += class_loss.item()
        running_bbox_loss += bbox_loss.item()
        
        # Log to TensorBoard
        if (i + 1) % log_step_interval == 0:
            step = epoch * len(dataloader) + i
            writer.add_scalar('Training/Loss', loss.item(), step)
            writer.add_scalar('Training/ClassLoss', class_loss.item(), step)
            writer.add_scalar('Training/BBoxLoss', bbox_loss.item(), step)
            
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(dataloader)}, "
                  f"Loss: {loss.item():.4f}, Class Loss: {class_loss.item():.4f}, "
                  f"BBox Loss: {bbox_loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    avg_class_loss = running_class_loss / len(dataloader)
    avg_bbox_loss = running_bbox_loss / len(dataloader)
    
    print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.4f}, "
          f"Avg Class Loss: {avg_class_loss:.4f}, "
          f"Avg BBox Loss: {avg_bbox_loss:.4f}")
    
    return avg_loss, avg_class_loss, avg_bbox_loss

def test(dataloader, model, class_loss_fn, bbox_loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_bbox_loss = 0.0
    
    # Lists to store predictions and ground truth
    all_y_preds = []
    all_y_trues = []
    all_bbox_preds = []
    all_bbox_trues = []
    
    with torch.no_grad():
        for images, targets, bboxes in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            bboxes = bboxes.to(device)
            
            # Forward pass
            class_preds, bbox_preds = model(images)
            
            # Calculate losses
            class_loss = class_loss_fn(class_preds, targets)
            bbox_loss = bbox_loss_fn(bbox_preds, bboxes)
            
            # Update total losses
            total_loss += class_loss.item()
            total_bbox_loss += bbox_loss.item()
            
            # Store predictions and ground truth
            all_y_preds.append(class_preds.argmax(1).cpu())
            all_y_trues.append(targets.cpu())
            all_bbox_preds.append(bbox_preds.cpu())
            all_bbox_trues.append(bboxes.cpu())
    
    # Concatenate all predictions and ground truth
    if all_y_preds:
        all_y_preds = torch.cat(all_y_preds)
        all_y_trues = torch.cat(all_y_trues)
        all_bbox_preds = torch.cat(all_bbox_preds)
        all_bbox_trues = torch.cat(all_bbox_trues)
    else:
        all_y_preds = torch.tensor([])
        all_y_trues = torch.tensor([])
        all_bbox_preds = torch.tensor([])
        all_bbox_trues = torch.tensor([])
    
    avg_loss = total_loss / max(len(dataloader), 1)
    avg_bbox_loss = total_bbox_loss / max(len(dataloader), 1)
    
    return avg_loss, avg_bbox_loss, all_y_preds, all_y_trues, all_bbox_preds, all_bbox_trues

def plot_predictions(images, labels, bboxes, idx_to_class, pred_labels=None, pred_bboxes=None, 
                    num_samples=8, save_path=None, figsize=(15, 15)):
    """
    Plot images with ground truth and predicted bounding boxes for traffic signs.
    
    Args:
        images (torch.Tensor): Batch of images
        labels (torch.Tensor): Ground truth labels
        bboxes (torch.Tensor): Ground truth bounding boxes (normalized format: [x, y, w, h])
        idx_to_class (dict): Mapping from class index to class name
        pred_labels (torch.Tensor, optional): Predicted labels
        pred_bboxes (torch.Tensor, optional): Predicted bounding boxes (normalized format)
        num_samples (int): Number of samples to plot
        save_path (str, optional): Path to save the plot
        figsize (tuple): Figure size
    """
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Take a subset of the data
    n_samples = min(num_samples, images.size(0))
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n_samples)))
    rows = cols = grid_size
    
    # Create a grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    for i in range(len(axs)):
        if i < n_samples:
            # Get image and convert to numpy
            img = images[i].cpu().clone()
            img = img * std + mean  # Denormalize
            img = img.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
            img = np.clip(img, 0, 1)  # Clip values to [0, 1]
            
            # Plot image
            axs[i].imshow(img)
            
            # Get true label and bbox
            true_label = idx_to_class[labels[i].item()]
            true_bbox = bboxes[i].cpu().numpy()
            
            # Convert normalized bbox to pixel coordinates
            h, w = img.shape[:2]
            true_x, true_y, true_w, true_h = true_bbox
            true_x = true_x * w
            true_y = true_y * h
            true_w = true_w * w
            true_h = true_h * h
            
            # Create a rectangle patch for true bbox
            true_rect = patches.Rectangle(
                (true_x, true_y), true_w, true_h, 
                linewidth=2, edgecolor='g', facecolor='none'
            )
            axs[i].add_patch(true_rect)
            
            # Add true label
            axs[i].text(
                true_x, true_y - 5, 
                f"True: {true_label}", 
                color='green', fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            # If predictions are provided, add them too
            if pred_labels is not None and pred_bboxes is not None:
                pred_label = idx_to_class[pred_labels[i].item()]
                pred_bbox = pred_bboxes[i].detach().cpu().numpy()
                
                # Convert normalized bbox to pixel coordinates
                pred_x, pred_y, pred_w, pred_h = pred_bbox
                pred_x = pred_x * w
                pred_y = pred_y * h
                pred_w = pred_w * w
                pred_h = pred_h * h
                
                # Create a rectangle patch for pred bbox
                pred_rect = patches.Rectangle(
                    (pred_x, pred_y), pred_w, pred_h, 
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                axs[i].add_patch(pred_rect)
                
                # Add pred label
                axs[i].text(
                    pred_x, pred_y - 20, 
                    f"Pred: {pred_label}", 
                    color='red', fontsize=9, 
                    bbox=dict(facecolor='white', alpha=0.8)
                )
            
            # Turn off axis
            axs[i].axis('off')
        else:
            # Hide unused subplots
            axs[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.close()

def inference_on_image(model, image_path, transform, idx_to_class, device='cpu'):
    """
    Run inference on a single image and return predictions
    
    Args:
        model: Trained model
        image_path: Path to the image
        transform: Image transformations to apply
        idx_to_class: Mapping from class index to class name
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence, bbox)
    """
    from PIL import Image
    from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
    
    # Load and prepare image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    # Create a dummy bbox for transformation (will be ignored)
    dummy_bbox = BoundingBoxes(
        [0, 0, 10, 10],  # Dummy values
        format="xywh", 
        canvas_size=(img_height, img_width)
    )
    
    img_tensor = TVImage(img)
    
    # Apply transformations
    transformed = transform(img_tensor, dummy_bbox)
    img_tensor, _ = transformed
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        class_preds, bbox_preds = model(img_tensor)
        
        # Get predicted class and confidence
        confidence, pred_idx = torch.max(torch.nn.functional.softmax(class_preds, dim=1), dim=1)
        pred_class = idx_to_class[pred_idx.item()]
        
        # Get predicted bounding box
        bbox = bbox_preds[0].cpu().numpy()  # [x, y, w, h] normalized
        
    return pred_class, confidence.item(), bbox