import torch
import os
import numpy as np
from sklearn.calibration import calibration_curve

from models.uncertainty_wrapper import MCDropoutWrapper
from metrics.classification_metrics import calculate_classification_metrics
from metrics.calibration_metrics import calculate_calibration_metrics
from explainability.gradcam import GradCAM
from explainability.saliency import SaliencyMap
from explainability.integrated_gradients import IntegratedGradients
from utils.visualization import plot_confusion_matrix, plot_calibration_curve, save_xai_overlay

class Evaluator:
    def __init__(self, model, dataloader, device, logger, save_dir, classes, config):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.classes = classes
        self.config = config
        
        self.uncertainty_method = config['uncertainty']['method']
        self.num_samples = config['uncertainty'].get('num_samples', 30)
        self.xai_method = config['explainability'].get('method', 'gradcam')

    def evaluate(self):
        self.logger.info("Starting comprehensive evaluation...")
        
        wrapper = MCDropoutWrapper(self.model, num_samples=self.num_samples)
        
        all_labels = []
        all_expected_probs = []
        all_entropy = []
        all_variance = []
        all_img_paths = []
        
        eval_start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        eval_end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if eval_start: eval_start.record()
        
        for images, labels, paths in self.dataloader:
            images = images.to(self.device)
            labels = labels.numpy()
            
            # Predict with uncertainty (Monte Carlo Dropout)
            predictions = wrapper.predict(images) # [samples, batch, classes]
            predictions = predictions.cpu()
            
            expected_p, entropy, variance = wrapper.get_uncertainty_metrics(predictions)
            
            all_labels.extend(labels)
            all_expected_probs.append(expected_p.numpy())
            all_entropy.extend(entropy.numpy())
            all_variance.append(variance.numpy())
            all_img_paths.extend(paths)
            
        if eval_end: 
            eval_end.record()
            torch.cuda.synchronize()
            latency = eval_start.elapsed_time(eval_end) / len(self.dataloader.dataset)
        else:
            latency = 0.0 # CPU timing simplified for now
            
        all_labels = np.array(all_labels)
        all_expected_probs = np.vstack(all_expected_probs)
        all_predictions = np.argmax(all_expected_probs, axis=1)
        all_confidences = np.max(all_expected_probs, axis=1)
        
        # 1. Classification Metrics
        class_metrics = calculate_classification_metrics(all_labels, all_predictions, all_expected_probs)
        self.logger.info("--- Classification Metrics ---")
        self.logger.info(f"Accuracy:  {class_metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {class_metrics['precision']:.4f}")
        self.logger.info(f"Recall:    {class_metrics['recall']:.4f}")
        self.logger.info(f"F1-Score:  {class_metrics['f1']:.4f}")
        self.logger.info(f"AUC Score: {class_metrics['auc']:.4f}")
        
        # Plot CM
        cm_path = os.path.join(self.save_dir, 'plots', 'confusion_matrix.png')
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plot_confusion_matrix(class_metrics['cm'], self.classes, cm_path)
        
        # 2. Calibration Metrics
        calib_metrics = calculate_calibration_metrics(all_expected_probs, all_labels)
        self.logger.info("--- Calibration Metrics ---")
        self.logger.info(f"ECE:            {calib_metrics['ece']:.4f}")
        self.logger.info(f"MCE:            {calib_metrics['mce']:.4f}")
        self.logger.info(f"NLL (Log Loss): {calib_metrics['nll']:.4f}")
        self.logger.info(f"Brier Score:    {calib_metrics['brier']:.4f}")
        
        avg_entropy = np.mean(all_entropy)
        avg_variance = np.mean(np.concatenate(all_variance))
        self.logger.info(f"Mean Predictive Entropy: {avg_entropy:.4f}")
        self.logger.info(f"Mean Predictive Variance: {avg_variance:.4f}")
        
        # Curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            all_labels == all_predictions, all_confidences, n_bins=10)
        calib_path = os.path.join(self.save_dir, 'plots', 'calibration_curve.png')
        plot_calibration_curve(fraction_of_positives, mean_predicted_value, calib_path)
        
        return {
            'classification': class_metrics,
            'calibration': calib_metrics,
            'latency_ms': latency,
            'avg_entropy': avg_entropy,
            'avg_variance': avg_variance
        }
        
    def generate_explanations(self, num_samples=5):
        self.logger.info(f"\nGenerating XAI heatmaps using {self.xai_method}...")
        
        if self.xai_method == 'gradcam':
            xai_tool = GradCAM(self.model, self.model.get_cam_layer())
        elif self.xai_method == 'saliency':
            xai_tool = SaliencyMap(self.model)
        elif self.xai_method == 'integrated_gradients':
            xai_tool = IntegratedGradients(self.model)
        else:
            self.logger.warning(f"Unknown XAI method: {self.xai_method}")
            return
            
        count = 0
        plots_dir = os.path.join(self.save_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        for images, labels, paths in self.dataloader:
            images = images.to(self.device)
            for i in range(images.size(0)):
                if count >= num_samples:
                    return
                
                img_tensor = images[i:i+1] # [1, C, H, W]
                target = labels[i].item()
                
                heatmap = xai_tool.generate(img_tensor, target_class=target)
                
                # Save overlay
                save_path = os.path.join(plots_dir, f"{self.xai_method}_sample_{count}.png")
                save_xai_overlay(img_tensor[0], heatmap, save_path)
                
                count += 1
