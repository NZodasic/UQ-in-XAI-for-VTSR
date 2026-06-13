import csv
import glob
import os
import shutil
import time
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, device, logger, save_dir, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.full_config = config
        self.config = config.get('training', {})
        self.start_epoch = 1
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_model_state_dict = None
        self.use_amp = bool(self.config.get('mixed_precision', False)) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        
        # CrossEntropyLoss configuration
        label_smoothing = self.config.get('label_smoothing', 0.0)
        use_class_weights = self.config.get('use_class_weights', False)
        
        weights = None
        if use_class_weights and hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'samples'):
            try:
                class_counts = {}
                for sample in train_loader.dataset.samples:
                    cls_id = sample[1]
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
                
                configured_num_classes = self.full_config.get('data', {}).get('num_classes')
                num_classes = int(configured_num_classes or (max(class_counts.keys()) + 1))
                invalid_classes = sorted(c for c in class_counts if c < 0 or c >= num_classes)
                if invalid_classes:
                    raise ValueError(
                        f"Found class ids outside configured num_classes={num_classes}: "
                        f"{invalid_classes}"
                    )

                total_samples = sum(class_counts.values())
                weights = torch.zeros(num_classes, dtype=torch.float)

                missing_classes = []
                for c in range(num_classes):
                    count = class_counts.get(c, 0)
                    if count > 0:
                        weights[c] = total_samples / (num_classes * count)
                    else:
                        missing_classes.append(c)

                if missing_classes:
                    self.logger.warning(
                        f"Training data is missing configured class ids: {missing_classes}. "
                        "Assigning zero weight to these classes. Consider fixing the train split or reducing data.num_classes."
                    )

                weight_min = weights[weights > 0].min().item() if torch.any(weights > 0) else 0.0
                self.logger.info(
                    "Using balanced class weights "
                    f"(min={weight_min:.4f}, max={weights.max().item():.4f}, "
                    f"mean={weights.mean().item():.4f})"
                )
                weights = weights.to(self.device)
            except ValueError:
                raise
            except Exception as e:
                self.logger.warning(f"Could not compute class weights: {e}")
                
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        self.eval_criterion = nn.CrossEntropyLoss()
        
        # Scheduler
        epochs = self.config.get('epochs', 60)
        warmup_epochs = self.config.get('warmup_epochs', 5)
        
        if self.config.get('scheduler') == 'cosine':
            cosine_epochs = max(1, epochs - warmup_epochs)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=cosine_epochs,
                eta_min=self.config.get('min_lr', 1e-6)
            )
            
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
                )
                self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                    self.optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler], 
                    milestones=[warmup_epochs]
                )
            else:
                self.scheduler = main_scheduler
        else:
            self.scheduler = None
            
        self.grad_clip = self.config.get('grad_clip', 'none')
        self.early_stopping_cfg = self.config.get('early_stopping', {})
        self.early_stopping_enabled = bool(self.early_stopping_cfg.get('enabled', True))
        self.early_stopping_monitor = self.early_stopping_cfg.get('monitor', 'val_accuracy')
        self.early_stopping_mode = self.early_stopping_cfg.get('mode')
        if self.early_stopping_mode is None:
            self.early_stopping_mode = 'min' if self.early_stopping_monitor == 'val_loss' else 'max'
        if self.early_stopping_mode not in {'min', 'max'}:
            raise ValueError("training.early_stopping.mode must be 'min' or 'max'.")
        self.patience = int(self.early_stopping_cfg.get('patience', self.config.get('patience', 12)))
        self.min_delta = float(self.early_stopping_cfg.get('min_delta', 0.0))
        self.restore_best_weights = bool(self.early_stopping_cfg.get('restore_best_weights', True))
        self.best_monitor_value = None
        self.log_every_batches = int(self.config.get('log_every_batches', 50))
        self.checkpoint_cfg = self.config.get('checkpoint', {})
        self.checkpoint_every = max(0, int(self.checkpoint_cfg.get('save_every_epochs', 1)))
        self.keep_last = int(self.checkpoint_cfg.get('keep_last', 3))
        self.models_dir = os.path.join(self.save_dir, 'models')
        self.checkpoint_dir = os.path.join(self.models_dir, 'checkpoints')
        self.metrics_csv_path = os.path.join(self.save_dir, 'epoch_metrics.csv')
        self.log_epoch_metrics = bool(self.config.get('log_epoch_metrics', False))
        self.tb_writer = self._create_tensorboard_writer()

    def _create_tensorboard_writer(self):
        tb_cfg = self.config.get('tensorboard', {})
        if not tb_cfg.get('enabled', True):
            return None

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.logger.warning(
                "TensorBoard logging is enabled but tensorboard is not installed. "
                "Install it with: pip install tensorboard"
            )
            return None

        log_dir = tb_cfg.get('log_dir', 'tensorboard')
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(self.save_dir, log_dir)
        os.makedirs(log_dir, exist_ok=True)
        self.logger.info(f"TensorBoard logs: {log_dir}")
        return SummaryWriter(log_dir=log_dir)

    def _close_tensorboard(self):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

    def _current_lr(self):
        if not self.optimizer.param_groups:
            return 0.0
        return float(self.optimizer.param_groups[0].get('lr', 0.0))

    def _cpu_state_dict(self):
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    def _move_optimizer_state_to_device(self):
        for state in self.optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _torch_load(self, checkpoint_path):
        try:
            return torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)

    def load_checkpoint(self, checkpoint_path):
        if not checkpoint_path:
            return
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = self._torch_load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if checkpoint.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self._move_optimizer_state_to_device()
            if self.scheduler is not None and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if self.use_amp and checkpoint.get('scaler_state_dict'):
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

            self.best_val_acc = float(checkpoint.get('best_val_acc', 0.0))
            self.best_monitor_value = checkpoint.get('best_monitor_value')
            if self.best_monitor_value is not None:
                self.best_monitor_value = float(self.best_monitor_value)
            self.epochs_no_improve = int(checkpoint.get('epochs_no_improve', 0))
            self.train_losses = list(checkpoint.get('train_losses', []))
            self.val_losses = list(checkpoint.get('val_losses', []))
            self.val_accuracies = list(checkpoint.get('val_accuracies', []))
            if self.best_monitor_value is None:
                if self.early_stopping_monitor in {'val_accuracy', 'val_acc', 'accuracy'} and self.val_accuracies:
                    self.best_monitor_value = max(self.val_accuracies)
                elif self.early_stopping_monitor == 'val_loss' and self.val_losses:
                    self.best_monitor_value = min(self.val_losses)
            self.best_model_state_dict = checkpoint.get('best_model_state_dict')

            completed_epoch = int(checkpoint.get('epoch', 0))
            partial_epoch = checkpoint.get('partial_epoch')
            self.start_epoch = int(partial_epoch) if partial_epoch else completed_epoch + 1

            os.makedirs(self.models_dir, exist_ok=True)
            if self.best_model_state_dict is not None:
                torch.save(self.best_model_state_dict, os.path.join(self.models_dir, 'best_model.pth'))

            self.logger.info(
                f"Loaded checkpoint {checkpoint_path}; resuming at epoch "
                f"{self.start_epoch} with best val acc {self.best_val_acc:.4f}."
            )
            return

        self.model.load_state_dict(checkpoint)
        self.best_model_state_dict = self._cpu_state_dict()
        self.logger.warning(
            f"Loaded weights from {checkpoint_path}, but optimizer/scheduler state was not found. "
            "Training will start from epoch 1."
        )

    def _checkpoint_payload(
        self,
        epoch,
        best_val_acc,
        epochs_no_improve,
        train_losses,
        val_losses,
        val_accuracies,
        partial_epoch=None
    ):
        payload = {
            'epoch': int(epoch),
            'partial_epoch': partial_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'best_model_state_dict': self.best_model_state_dict,
            'best_val_acc': float(best_val_acc),
            'best_monitor_value': self.best_monitor_value,
            'epochs_no_improve': int(epochs_no_improve),
            'train_losses': list(train_losses),
            'val_losses': list(val_losses),
            'val_accuracies': list(val_accuracies),
            'config': self.full_config,
            'rng_state': {
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }
        }
        return payload

    def _prune_epoch_checkpoints(self):
        if self.keep_last <= 0:
            return
        checkpoint_paths = sorted(glob.glob(os.path.join(self.checkpoint_dir, 'epoch_*.pth')))
        stale_paths = checkpoint_paths[:-self.keep_last]
        for path in stale_paths:
            try:
                os.remove(path)
            except OSError as exc:
                self.logger.warning(f"Could not remove old checkpoint {path}: {exc}")

    def _save_checkpoint(
        self,
        epoch,
        best_val_acc,
        epochs_no_improve,
        train_losses,
        val_losses,
        val_accuracies,
        is_best=False,
        tag=None,
        partial_epoch=None
    ):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = tag if tag else f"epoch_{int(epoch):03d}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        payload = self._checkpoint_payload(
            epoch=epoch,
            best_val_acc=best_val_acc,
            epochs_no_improve=epochs_no_improve,
            train_losses=train_losses,
            val_losses=val_losses,
            val_accuracies=val_accuracies,
            partial_epoch=partial_epoch
        )
        torch.save(payload, checkpoint_path)

        latest_path = os.path.join(self.checkpoint_dir, 'latest.pth')
        shutil.copy2(checkpoint_path, latest_path)
        if is_best:
            shutil.copy2(checkpoint_path, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        if not tag:
            self._prune_epoch_checkpoints()
        return checkpoint_path

    def _checkpoints_enabled(self):
        return self.checkpoint_every > 0

    def _should_save_checkpoint(self, epoch, is_best, should_stop):
        if not self._checkpoints_enabled():
            return False
        if is_best or should_stop:
            return True
        return epoch % self.checkpoint_every == 0

    def _log_epoch_metrics(self, metrics):
        if not self.log_epoch_metrics and self.tb_writer is None:
            return

        fieldnames = [
            'epoch',
            'train_loss',
            'val_loss',
            'val_accuracy',
            'best_val_accuracy',
            'lr',
            'next_lr',
            'epochs_no_improve',
            'is_best',
            'checkpoint_path',
            'epoch_seconds'
        ]
        if self.log_epoch_metrics:
            file_exists = os.path.exists(self.metrics_csv_path)
            with open(self.metrics_csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({key: metrics.get(key) for key in fieldnames})

        if self.tb_writer is None:
            return
        epoch = metrics['epoch']
        self.tb_writer.add_scalar('Loss/train', metrics['train_loss'], epoch)
        self.tb_writer.add_scalar('Loss/val', metrics['val_loss'], epoch)
        self.tb_writer.add_scalar('Accuracy/val', metrics['val_accuracy'], epoch)
        self.tb_writer.add_scalar('Accuracy/best_val', metrics['best_val_accuracy'], epoch)
        self.tb_writer.add_scalar('LearningRate/current', metrics['lr'], epoch)
        self.tb_writer.add_scalar('LearningRate/next', metrics['next_lr'], epoch)
        self.tb_writer.add_scalar('EarlyStopping/epochs_no_improve', metrics['epochs_no_improve'], epoch)
        self.tb_writer.add_scalar('Checkpoint/is_best', int(metrics['is_best']), epoch)
        self.tb_writer.flush()

    def _get_monitor_value(self, val_loss, val_acc):
        if self.early_stopping_monitor in {'val_accuracy', 'val_acc', 'accuracy'}:
            return val_acc
        if self.early_stopping_monitor == 'val_loss':
            return val_loss
        raise ValueError(
            "training.early_stopping.monitor must be 'val_accuracy' or 'val_loss'."
        )

    def _is_improvement(self, current_value, best_value):
        if best_value is None:
            return True
        if self.early_stopping_mode == 'min':
            return current_value < best_value - self.min_delta
        return current_value > best_value + self.min_delta
        
    def train(self, epochs):
        best_val_acc = self.best_val_acc
        best_monitor_value = self.best_monitor_value
        epochs_no_improve = self.epochs_no_improve
        train_losses = list(self.train_losses)
        val_losses = list(self.val_losses)
        val_accuracies = list(self.val_accuracies)

        os.makedirs(self.models_dir, exist_ok=True)

        if self.start_epoch > epochs:
            self.logger.warning(
                f"Checkpoint already reached epoch {self.start_epoch - 1}; "
                f"requested epochs={epochs}. Skipping training."
            )
            self._close_tensorboard()
            return train_losses, val_losses, val_accuracies

        current_epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, epochs + 1):
                current_epoch = epoch
                epoch_start_time = time.time()
                lr = self._current_lr()
                self.logger.info(f"Starting epoch {epoch}/{epochs} ({len(self.train_loader)} train batches).")
                self.model.train()
                running_loss = 0.0

                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
                for batch_idx, (inputs, labels) in enumerate(pbar, start=1):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        log_loss = self.eval_criterion(outputs, labels)

                    with torch.no_grad():
                        log_loss_value = log_loss.detach()

                    self.scaler.scale(loss).backward()

                    if self.grad_clip != 'none':
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    running_loss += log_loss_value.item() * inputs.size(0)
                    pbar.set_postfix({
                        'loss': f"{log_loss_value.item():.4f}",
                        'opt_loss': f"{loss.item():.4f}"
                    })

                    if self.log_every_batches > 0 and (
                        batch_idx == 1
                        or batch_idx % self.log_every_batches == 0
                        or batch_idx == len(self.train_loader)
                    ):
                        self.logger.info(
                            f"Epoch {epoch:03d} train batch {batch_idx}/{len(self.train_loader)} "
                            f"| loss: {log_loss_value.item():.4f} | opt_loss: {loss.item():.4f}"
                        )

                epoch_train_loss = running_loss / len(self.train_loader.dataset)
                train_losses.append(epoch_train_loss)

                self.logger.info(f"Starting validation for epoch {epoch}/{epochs}.")
                val_loss, val_acc = self._validate()
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

                self.logger.info(
                    f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

                if self.scheduler:
                    self.scheduler.step()
                next_lr = self._current_lr()

                monitor_value = self._get_monitor_value(val_loss, val_acc)
                is_best_for_checkpoint = val_acc > best_val_acc
                is_improvement = self._is_improvement(monitor_value, best_monitor_value)
                if is_best_for_checkpoint:
                    best_val_acc = val_acc
                    self.best_model_state_dict = self._cpu_state_dict()
                    torch.save(self.best_model_state_dict, os.path.join(self.models_dir, 'best_model.pth'))
                    self.logger.info(f"--> Saved new best model (Acc: {best_val_acc:.4f})")

                if is_improvement:
                    best_monitor_value = monitor_value
                    epochs_no_improve = 0
                    self.best_model_state_dict = self._cpu_state_dict()
                    torch.save(self.best_model_state_dict, os.path.join(self.models_dir, 'best_model.pth'))
                    if not is_best_for_checkpoint:
                        self.logger.info(
                            f"--> Early-stopping monitor improved "
                            f"({self.early_stopping_monitor}: {monitor_value:.4f})"
                        )
                else:
                    epochs_no_improve += 1

                self.best_monitor_value = best_monitor_value
                should_stop = (
                    self.early_stopping_enabled
                    and self.patience > 0
                    and epochs_no_improve >= self.patience
                )
                checkpoint_path = None
                if self._should_save_checkpoint(epoch, is_best_for_checkpoint, should_stop):
                    checkpoint_path = self._save_checkpoint(
                        epoch=epoch,
                        best_val_acc=best_val_acc,
                        epochs_no_improve=epochs_no_improve,
                        train_losses=train_losses,
                        val_losses=val_losses,
                        val_accuracies=val_accuracies,
                        is_best=is_best_for_checkpoint
                    )
                    self.logger.info(f"Saved checkpoint: {checkpoint_path}")

                self._log_epoch_metrics({
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'best_val_accuracy': best_val_acc,
                    'lr': lr,
                    'next_lr': next_lr,
                    'epochs_no_improve': epochs_no_improve,
                    'is_best': is_best_for_checkpoint,
                    'checkpoint_path': checkpoint_path,
                    'epoch_seconds': time.time() - epoch_start_time
                })

                self.start_epoch = epoch + 1
                self.best_val_acc = best_val_acc
                self.epochs_no_improve = epochs_no_improve
                self.train_losses = train_losses
                self.val_losses = val_losses
                self.val_accuracies = val_accuracies

                if should_stop:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs. "
                        f"Best {self.early_stopping_monitor}: {best_monitor_value:.4f}; "
                        f"best validation accuracy: {best_val_acc:.4f}."
                    )
                    if self.restore_best_weights and self.best_model_state_dict is not None:
                        self.model.load_state_dict(self.best_model_state_dict)
                        torch.save(self.best_model_state_dict, os.path.join(self.models_dir, 'best_model.pth'))
                        self.logger.info("Restored best model weights after early stopping.")
                    break
        except KeyboardInterrupt:
            completed_epoch = len(val_accuracies)
            partial_epoch = current_epoch if current_epoch > completed_epoch else None
            if self._checkpoints_enabled():
                interrupt_path = self._save_checkpoint(
                    epoch=completed_epoch,
                    best_val_acc=best_val_acc,
                    epochs_no_improve=epochs_no_improve,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    val_accuracies=val_accuracies,
                    tag='interrupt_latest.pth',
                    partial_epoch=partial_epoch
                )
                self.logger.warning(f"Interrupted; saved checkpoint before exit: {interrupt_path}")
            else:
                self.logger.warning("Interrupted; checkpoint saving is disabled.")
            raise
        finally:
            self._close_tensorboard()

        return train_losses, val_losses, val_accuracies

    def _validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.eval_criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc
