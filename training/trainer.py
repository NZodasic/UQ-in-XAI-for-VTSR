import torch
import torch.nn as nn
import os
import time

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, 
                 device, logger, save_dir, patience=10):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.patience = patience
        
    def train(self, epochs):
        self.logger.info("Starting training loop...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Train step
            self.model.train()
            running_loss = 0.0
            
            for images, labels, _ in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                
            epoch_train_loss = running_loss / len(self.train_loader.dataset)
            train_losses.append(epoch_train_loss)
            
            # Validation step
            self.model.eval()
            running_val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for images, labels, _ in self.val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    running_val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    
            epoch_val_loss = running_val_loss / len(self.val_loader.dataset)
            epoch_val_acc = correct / len(self.val_loader.dataset)
            
            val_losses.append(epoch_val_loss)
            val_accuracies.append(epoch_val_acc)
            
            elapsed = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} " 
                             f"| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Time: {elapsed:.1f}s")
                             
            # Checkpoint saving & early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                
                checkpoint_path = os.path.join(self.save_dir, 'models')
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(checkpoint_path, 'best_model.pth'))
                self.logger.info(f"Saved new best model at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        return train_losses, val_losses, val_accuracies
