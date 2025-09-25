import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os



def train_one_epoch(model, dataloader, optimizer, criterion, device, 
                    scaler=None, grad_accum_steps=1, aurora=True, north = False):
    model.train()
    total_loss = 0
    batch_count = len(dataloader)
    optimizer.zero_grad()
    here = os.path.dirname(os.path.abspath(__file__))

    
    with tqdm(dataloader, desc="Training") as pbar:
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)
            
            if aurora:
                lat_mask = targets[:,-1]
                targets = targets[:,:-1]
                
            


            # Mixed precision training
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    # Check for NaNs and handle properly (log or raise)
                    if torch.isnan(outputs).any():
                        print("Warning: NaNs detected in model outputs!")
                        # Optionally: raise RuntimeError or handle differently
                        outputs = torch.clamp(outputs, -20, 20) 
                        
                    if aurora == True:
                        outputs = outputs.squeeze(dim = 1)
                        targets = targets.squeeze(dim = 2)
                    else:
                        outputs = outputs.squeeze()
                    
                    # outputs = outputs* lat_mask.unsqueeze(1)  # Now shape [64, 3, 52, 92]
                    # Masked loss
                    loss = criterion(outputs, targets.squeeze())
                    loss = loss / grad_accum_steps  # Normalize loss for accumulation
                
                # Accumulate scaled gradients
                with torch.autograd.set_detect_anomaly(True):
                    scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == batch_count:
                    # Unscale gradients for potential clipping
                    scaler.unscale_(optimizer)
                    
                    # Optional: gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard training path without mixed precision
                outputs = model(images)
                outputs = outputs.squeeze()
                
                
                lat_mask = pd.read_csv(os.path.join(here, "..", "model_comparisons", "latitude.csv")).to_numpy()[:52,1:]
                lat_mask[np.isnan(lat_mask)] = 0
                lat_mask[lat_mask != 0] = 1
                if north:
                    lat_mask = np.flipud(lat_mask)
                lat_mask = torch.tensor(lat_mask, dtype=targets.dtype, device=targets.device)

                targets = targets * lat_mask
                outputs = outputs * lat_mask
                
                
                targets = targets*lat_mask
                outputs = targets*lat_mask
                loss = criterion(outputs, targets)
                loss = loss / grad_accum_steps

                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == batch_count:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # For progress tracking (use original loss value)
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(loss=loss.item() * grad_accum_steps)
    
    return total_loss / batch_count


def evaluate(model, dataloader, criterion, device, aurora=True):
    model.eval()
    running_loss = 0.0
    
    if aurora:  # Segmentation metrics
        dice_sum = 0.0
        mcc_sum = 0.0
    else:  # Regression metrics
        mae_sum = 0.0
        mse_sum = 0.0
        r2_sum = 0.0
    


    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)
            
            if aurora:
                lat_mask = targets[:,-1]
                targets = targets[:,:-1]
                
            outputs = model(images)
            if aurora:  # Segmentation task
                # outputs = outputs.squeeze(1)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
                # Apply sigmoid to get probabilities
                
                
                
                probs = torch.sigmoid(outputs)
                # Convert to binary predictions using 0.5 threshold
                preds = (probs > 0.5).float()
                
                # preds = preds* lat_mask.unsqueeze(1)  # Now shape [64, 3, 52, 92]
                # Calculate Dice coefficient
                epsilon = 1e-6
                intersection = (preds * targets).sum()
                dice = (2. * intersection) / (preds.sum() + targets.sum() + epsilon)
                dice_sum += dice.item()
                
                # Calculate Matthews Correlation Coefficient (MCC)
                preds_flat = preds.flatten()
                targets_flat = targets.flatten()
                
                # Calculate confusion matrix components
                tp = (preds_flat * targets_flat).sum().float()
                tn = ((1 - preds_flat) * (1 - targets_flat)).sum().float()
                fp = (preds_flat * (1 - targets_flat)).sum().float()
                fn = ((1 - preds_flat) * targets_flat).sum().float()
                
                # Calculate MCC
                numerator = tp * tn - fp * fn
                denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                
                if denominator == 0:
                    mcc = torch.tensor(0.0)
                else:
                    mcc = numerator / denominator
                
                mcc_sum += mcc.item()
                
            else:  # Regression task
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                
                # Calculate regression metrics
                mae = torch.mean(torch.abs(outputs - targets))
                mse = torch.mean((outputs - targets) ** 2)
                
                # Calculate R² (coefficient of determination)
                ss_res = torch.sum((targets - outputs) ** 2)
                ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
                r2_vals = []
                for i in range(targets.shape[0]):
                    t = targets[i].flatten()
                    o = outputs[i].flatten()
                    ss_res = torch.sum((t - o) ** 2)
                    ss_tot = torch.sum((t - torch.mean(t)) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-8)
                    r2_vals.append(r2.item())
                
                r2_batch = sum(r2_vals) / len(r2_vals)
                mae_sum += mae.item()
                mse_sum += mse.item()
                r2_sum += r2_batch

    # Calculate average metrics
    epoch_loss = running_loss / len(dataloader)
    
    if aurora:  # Segmentation metrics
        epoch_dice = dice_sum / len(dataloader)
        epoch_mcc = mcc_sum / len(dataloader)
        print(f"Validation Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, MCC: {epoch_mcc:.4f}")
        return epoch_loss, epoch_dice, epoch_mcc
    else:  # Regression metrics
        epoch_mae = mae_sum / len(dataloader)
        epoch_mse = mse_sum / len(dataloader)
        epoch_r2 = r2_sum / len(dataloader)
        epoch_rmse = torch.sqrt(torch.tensor(epoch_mse)).item()
        print(f"Validation Loss: {epoch_loss:.4f}, MAE: {epoch_mae:.4f}, MSE: {epoch_mse:.4f}, RMSE: {epoch_rmse:.4f}, R²: {epoch_r2:.4f}")
        return epoch_loss, epoch_mae, epoch_mse, epoch_rmse, epoch_r2


def run_training(model, train_loader, val_loader, device, 
                  epochs=10, check_val=True,
                  lr=1e-3, weight_decay=1e-4,
                  patience=5, grad_accum_steps=1, aurora=True, north = False):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    if aurora:  # Segmentation task
        criterion = nn.BCEWithLogitsLoss()
    else:  # Regression task
        criterion = nn.MSELoss()  
    
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    if aurora:
        best_dice = 0.0
        best_mcc = -1.0
    else:
        best_mae = float('inf')
        best_r2 = -float('inf')
    
    no_improve_epochs = 0
    
    if aurora:
        history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_mcc': []}
    else:
        history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_mse': [], 'val_rmse': [], 'val_r2': []}
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train with mixed precision and gradient accumulation
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, 
            scaler=scaler, grad_accum_steps=grad_accum_steps, aurora=aurora, north = north
        )
        history['train_loss'].append(train_loss)
        
        if check_val:
            if aurora:  # Segmentation evaluation
                val_loss, val_dice, val_mcc = evaluate(model, val_loader, criterion, device, aurora=aurora)
                history['val_loss'].append(val_loss)
                history['val_dice'].append(val_dice)
                history['val_mcc'].append(val_mcc)
                
                # Update learning rate based on validation loss
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val MCC: {val_mcc:.4f}")
                
                # Save model based on MCC since it's better for imbalanced datasets
                if val_dice > best_dice:
                    best_mcc = val_mcc
                    best_dice = val_dice
                    best_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_dice': val_dice,
                        'val_mcc': val_mcc,
                        'task_type': 'segmentation'
                    }, 'best_seg_model.pth')
                    print(f"✅ Saved Best Model (Best Dice: {best_dice:.4f})")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    
            else:  # Regression evaluation
                val_loss, val_mae, val_mse, val_rmse, val_r2 = evaluate(model, val_loader, criterion, device, aurora=aurora)
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                history['val_mse'].append(val_mse)
                history['val_rmse'].append(val_rmse)
                history['val_r2'].append(val_r2)
                
                # Update learning rate based on validation loss
                scheduler.step(val_loss)
                
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}")
                
                # Save model based on R² (higher is better) or MAE (lower is better)
                if val_r2 > best_r2:
                    best_r2 = val_r2
                    best_mae = val_mae
                    best_loss = val_loss
                    
                    save_path = 'best_reg_model.pth'
                    if north:
                        save_path = 'best_north.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_mae': val_mae,
                        'val_mse': val_mse,
                        'val_rmse': val_rmse,
                        'val_r2': val_r2,
                        'task_type': 'regression'
                    }, save_path)
                    print(f"✅ Saved Best Model (Best R²: {best_r2:.4f})")
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                
            # Early stopping
            if no_improve_epochs >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        else:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
    
    print("✅ Training Completed")
    if aurora:
        print(f"Best Dice: {best_dice:.4f}, Best MCC: {best_mcc:.4f}, Best Loss: {best_loss:.4f}")
    else:
        print(f"Best MAE: {best_mae:.4f}, Best R²: {best_r2:.4f}, Best Loss: {best_loss:.4f}")
    
    return history


def visualize_predictions(model, dataloader, device, aurora=True, north = False):
    model.eval()
    here = os.path.dirname(os.path.abspath(__file__))

    with torch.no_grad():
        for idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            
            
            outputs = model(images)
            
            
            if aurora:  # Segmentation visualization
                targets = targets[:,0].to(device)
                outputs = outputs.squeeze(1)
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                
                # Calculate Dice coefficient
                epsilon = 1e-6
                intersection = (preds * targets).sum()
                dice = (2. * intersection) / (preds.sum() + targets.sum() + epsilon)
                dice_sum = dice.item()
                print('Test set Dice: ', round(dice_sum,2))
                
                
                for i in range(images.size(0)):
                    image = images[i].cpu().squeeze()
                    target = targets[i].cpu().squeeze()
                    pred = preds[i].cpu().squeeze()

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                    axs[0].imshow(image[0], cmap='gray')
                    axs[0].set_title('Input Image')
                    axs[1].imshow(target, cmap='gray')
                    axs[1].set_title('Ground Truth Mask')
                    axs[2].imshow(pred, cmap='gray')
                    axs[2].set_title('Predicted Mask')

                    for ax in axs:
                        ax.axis('off')

                    plt.tight_layout()
                    plt.show()
                    
            else:  # Regression visualization
            
                criterion = nn.MSELoss()  
                loss = criterion(outputs.detach().cpu(),targets).item()
                ss_res = torch.sum((targets - outputs.detach().cpu()) ** 2)
                ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))  # Add small epsilon to avoid division by zero
                print('Test loss: ' , round(loss, 2))
                print('Test R2: ', round(r2.item(),2))
                for i in range(images.size(0)):
                    image = images[i].cpu().squeeze()
                    kp = torch.max(image[-4:, 0 , 0])
                    # if kp <4:
                    #     continue
                    target = targets[i].cpu().squeeze()
                    pred = outputs[i].cpu().squeeze()
                    lat_mask = pd.read_csv(os.path.join(here, "..", "model_comparisons", "latitude.csv")).to_numpy()[:52,1:]
                    lat_mask[np.isnan(lat_mask)] = 0
                    lat_mask[lat_mask != 0] = 1

                    if north:
                        lat_mask = np.flipud(lat_mask)
                        
                    target = target * lat_mask
                    pred = pred * lat_mask
                    
                    fig, axs = plt.subplots(1, 3, figsize=(24, 5))
                    plt.suptitle(f'Dayglow Prediction. Max Kp_6 = {kp* 1.2907782 + 1.21870199}')
                    # Input image
                    im = axs[0].imshow(image[0], cmap='gray')
                    axs[0].set_title('Input Image')
                    plt.colorbar(im, ax=axs[0])

                    # Ground truth
                    c = int(np.random.rand()//.33333)
                    im1 = axs[1].imshow(target[c], cmap='viridis')
                    axs[1].set_title('Ground Truth')
                    plt.colorbar(im1, ax=axs[1])
                    
                    # Prediction
                    im2 = axs[2].imshow(pred[c], cmap='viridis')
                    axs[2].set_title('Prediction')
                    plt.colorbar(im1, ax=axs[2])

                    for ax in axs:
                        ax.axis('off')

                    plt.tight_layout()
                    plt.show()
                    
                    
                    
    model.train()

        
def load_model(model, model_path, device):
    """
    Load saved model weights into your model.
    
    Args:
        model: Your PyTorch model instance
        model_path: Path to the saved weights file (.pth)
        device: The device to load the model onto ('cuda' or 'cpu')
    
    Returns:
        The model with loaded weights
    """
    # Load weights from file
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Handle different saving formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # If you saved with the complete checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        # If you saved just the model state dict directly
        model.load_state_dict(checkpoint)
        print("Loaded model weights")
    
    # Set model to evaluation mode
    model.eval()
    return model    
    
    
    
    
def print_loss(model_path = r"C:\Users\dogbl\OneDrive\Desktop\England Research\Aurora\jordan_aurora\deep_learning\best_reg_model.pth"):
    checkpoint = torch.load(model_path, map_location="cuda")
    
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Training Loss: {train_loss}")
    print(f"Validation Loss: {val_loss}")
    print(f"Validation MAE: {checkpoint['val_mae']}")
    print(f"Validation R2: {checkpoint['val_r2']}")

    
    
    
    
