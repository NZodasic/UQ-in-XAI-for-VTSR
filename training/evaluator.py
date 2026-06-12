import os
import time
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from metrics.calibration_metrics import calculate_calibration_metrics
from metrics.classification_metrics import calculate_classification_metrics
from models.uncertainty_wrapper import MCDropoutWrapper

def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    return heatmap / (heatmap.max() + 1e-8)

class Evaluator:
    def __init__(self, model, dataloader, device, logger, save_dir, classes, config):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.logger = logger
        self.save_dir = save_dir
        self.classes = classes
        self.config = config

    def _target_probability(self, input_tensor, target_class):
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
        return float(probs[0, int(target_class)].item())

    def _calculate_fidelity(self, input_tensor, target_class, heatmap, base_confidence):
        xai_cfg = self.config.get('explainability', {})
        mask_fraction = float(xai_cfg.get('fidelity_mask_fraction', 0.2))
        mask_fraction = min(max(mask_fraction, 0.01), 0.9)

        threshold = np.quantile(heatmap, 1.0 - mask_fraction)
        mask_np = heatmap >= threshold
        mask = torch.from_numpy(mask_np).to(input_tensor.device).bool().view(1, 1, *mask_np.shape)

        masked_input = input_tensor.clone()
        masked_input = masked_input.masked_fill(mask, 0.0)
        masked_confidence = self._target_probability(masked_input, target_class)
        return float(base_confidence - masked_confidence)

    def _calculate_stability(self, heatmap, noisy_heatmap):
        heatmap = normalize_heatmap(heatmap)
        noisy_heatmap = normalize_heatmap(noisy_heatmap)
        difference = float(np.mean(np.abs(heatmap - noisy_heatmap)))
        return max(0.0, 1.0 - difference)

    def _noisy_input(self, input_tensor):
        xai_cfg = self.config.get('explainability', {})
        noise_std = float(xai_cfg.get('stability_noise_std', 0.02))
        return input_tensor + torch.randn_like(input_tensor) * noise_std

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        latencies = []
        all_probabilities = []
        all_labels = []
        all_predictions = []
        all_entropy = []
        all_variance = []

        uncertainty_cfg = self.config.get('uncertainty', {})
        uncertainty_method = uncertainty_cfg.get('method', 'none').lower()
        use_mc_dropout = uncertainty_method == 'mc_dropout'
        num_uncertainty_samples = max(1, int(uncertainty_cfg.get('num_samples', 20)))
        mc_wrapper = None
        if use_mc_dropout:
            mc_wrapper = MCDropoutWrapper(
                self.model,
                num_samples=num_uncertainty_samples
            )
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.dataloader, desc="Evaluating"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                if use_mc_dropout:
                    mc_predictions = mc_wrapper.predict(inputs)
                    probabilities, entropy, variance = mc_wrapper.get_uncertainty_metrics(mc_predictions)
                    all_entropy.append(entropy.detach().cpu().numpy())
                    all_variance.append(variance.mean(dim=1).detach().cpu().numpy())
                else:
                    outputs = self.model(inputs)
                    probabilities = torch.softmax(outputs, dim=1)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()

                latencies.append((end_time - start_time) / inputs.size(0))
                
                _, predicted = probabilities.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_probabilities.append(probabilities.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
                all_predictions.append(predicted.detach().cpu().numpy())
                
        accuracy = correct / max(total, 1)
        avg_latency_ms = np.mean(latencies) * 1000 if latencies else 0.0
        probabilities_np = np.concatenate(all_probabilities, axis=0) if all_probabilities else np.empty((0, 0))
        labels_np = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
        predictions_np = np.concatenate(all_predictions, axis=0) if all_predictions else np.empty((0,), dtype=np.int64)
        classification = (
            calculate_classification_metrics(labels_np, predictions_np, probabilities_np)
            if total > 0 else {'accuracy': accuracy}
        )
        classification['accuracy'] = accuracy
        calibration = calculate_calibration_metrics(probabilities_np, labels_np) if total > 0 else {}
        uncertainty = {}
        if use_mc_dropout and all_entropy and all_variance:
            entropy_np = np.concatenate(all_entropy, axis=0)
            variance_np = np.concatenate(all_variance, axis=0)
            uncertainty = {
                'method': 'mc_dropout',
                'num_samples': num_uncertainty_samples,
                'predictive_entropy_mean': float(np.mean(entropy_np)),
                'predictive_entropy_std': float(np.std(entropy_np)),
                'probability_variance_mean': float(np.mean(variance_np)),
                'probability_variance_std': float(np.std(variance_np))
            }
        
        results = {
            'classification': classification,
            'calibration': calibration,
            'latency_ms': avg_latency_ms
        }
        if uncertainty:
            results['uncertainty'] = uncertainty
        return results

    def generate_explanations(self, num_samples=5):
        xai_cfg = self.config.get('explainability', {})
        xai_method = xai_cfg.get('method', 'gradcam').lower()

        if xai_method == 'integrated_gradients':
            return self._generate_integrated_gradients(num_samples, xai_cfg)

        if xai_method == 'saliency':
            return self._generate_saliency(num_samples, xai_cfg)

        if xai_method != 'gradcam':
            self.logger.warning(f"Unsupported XAI method '{xai_method}', skipping explanations.")
            return {}

        try:
            from explainability.gradcam import MCGradCAM
        except ImportError:
            self.logger.warning("GradCAM not found, skipping explanations.")
            return {}
            
        self.logger.info("Generating UQ-enhanced Grad-CAM explanations...")
        
        # Find target layer for GradCAM based on model type
        target_layer = None
        if hasattr(self.model, 'get_cam_layer'):
            target_layer = self.model.get_cam_layer()
        elif hasattr(self.model, 'features'):
            target_layer = self.model.features[-1]
        elif hasattr(self.model, 'layer4'):
            target_layer = self.model.layer4[-1]
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'features'):
            target_layer = self.model.model.features[-1]
        elif hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'layer4'):
            target_layer = self.model.backbone.layer4[-1]
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'layer4'):
            target_layer = self.model.base_model.layer4[-1]
            
        if target_layer is None:
            self.logger.warning("Could not automatically find target layer for GradCAM. Skipping.")
            return {}

        cam_method = xai_cfg.get('variant', 'gradcam') # 'gradcam', 'gradcam++', 'eigencam'
        mc_samples = xai_cfg.get('mc_samples', 10)
        compute_xai_metrics = bool(xai_cfg.get('compute_metrics', True))
        
        cam_gen = MCGradCAM(self.model, target_layer, cam_method=cam_method, num_samples=mc_samples)
        
        save_path = os.path.join(self.save_dir, 'xai')
        os.makedirs(save_path, exist_ok=True)
        
        samples_done = 0
        generation_times = []
        fidelity_scores = []
        stability_scores = []
        self.model.eval()
        
        try:
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                for i in range(inputs.size(0)):
                    if samples_done >= num_samples:
                        break
                        
                    single_input = inputs[i:i+1]
                    true_label = labels[i].item()

                    with torch.no_grad():
                        outputs = self.model(single_input)
                        probs = torch.softmax(outputs, dim=1)
                        conf, predicted_label = probs.max(1)
                        conf = conf.item()
                        predicted_label = predicted_label.item()

                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.time()
                    mean_heatmap, std_heatmap = cam_gen.generate(single_input, target_class=predicted_label)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    generation_times.append(time.time() - start_time)

                    if compute_xai_metrics:
                        fidelity_scores.append(
                            self._calculate_fidelity(single_input, predicted_label, mean_heatmap, conf)
                        )
                        noisy_heatmap, _ = cam_gen.generate(
                            self._noisy_input(single_input),
                            target_class=predicted_label
                        )
                        stability_scores.append(self._calculate_stability(mean_heatmap, noisy_heatmap))
                    
                    # Unnormalize image for visualization
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                    
                    img_tensor = single_input[0] * std + mean
                    img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
                    img_np = np.clip(img_np, 0, 1)
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    # Mean Heatmap Overlay
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * mean_heatmap), cv2.COLORMAP_JET)
                    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                    cam_overlay = heatmap_color * 0.4 + img_np
                    cam_overlay = np.clip(cam_overlay, 0, 255).astype(np.uint8)

                    # Uncertainty Heatmap Overlay (using inferno or similar to distinguish)
                    # Normalize std_heatmap for visualization
                    std_norm = normalize_heatmap(std_heatmap)
                    std_color = cv2.applyColorMap(np.uint8(255 * std_norm), cv2.COLORMAP_HOT)
                    std_color = cv2.cvtColor(std_color, cv2.COLOR_BGR2RGB)
                    std_overlay = std_color * 0.4 + img_np
                    std_overlay = np.clip(std_overlay, 0, 255).astype(np.uint8)

                    plt.figure(figsize=(25, 5))
                    
                    # 1. Original Image
                    plt.subplot(1, 5, 1)
                    plt.imshow(img_np)
                    class_name = self.classes.get(true_label, f"Class {true_label}")
                    plt.title(f"Original: {class_name}")
                    plt.axis('off')
                    
                    # 2. Prediction Info
                    plt.subplot(1, 5, 2)
                    pred_name = self.classes.get(predicted_label, f"Class {predicted_label}")
                    color = 'green' if predicted_label == true_label else 'red'
                    plt.text(0.5, 0.6, f"Pred: {pred_name}", fontsize=12, ha='center', color=color)
                    plt.text(0.5, 0.4, f"Conf: {conf:.4f}", fontsize=12, ha='center')
                    plt.title("Prediction Details")
                    plt.axis('off')
                    
                    # 3. Mean Grad-CAM
                    plt.subplot(1, 5, 3)
                    plt.imshow(cam_overlay)
                    plt.title(f"Mean {cam_method.upper()}")
                    plt.axis('off')

                    # 4. Uncertainty (Std)
                    plt.subplot(1, 5, 4)
                    plt.imshow(std_overlay)
                    plt.title("Explanation Uncertainty")
                    plt.axis('off')
                    
                    # 5. UQ-Aware Saliency (Fusion)
                    # Regions that are both high saliency and LOW uncertainty are more reliable
                    plt.subplot(1, 5, 5)
                    std_norm = normalize_heatmap(std_heatmap)
                    reliable_saliency = mean_heatmap * (1.0 - std_norm)
                    reliable_saliency = normalize_heatmap(reliable_saliency)
                    
                    reliable_color = cv2.applyColorMap(np.uint8(255 * reliable_saliency), cv2.COLORMAP_VIRIDIS)
                    reliable_color = cv2.cvtColor(reliable_color, cv2.COLOR_BGR2RGB)
                    reliable_overlay = reliable_color * 0.4 + img_np
                    reliable_overlay = np.clip(reliable_overlay, 0, 255).astype(np.uint8)
                    
                    plt.imshow(reliable_overlay)
                    plt.title("Reliable Saliency (Fusion)")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f'uq_gradcam_sample_{samples_done}.png'))
                    plt.close()
                    
                    samples_done += 1

                if samples_done >= num_samples:
                    break
        finally:
            cam_gen.remove_hooks()

        avg_time_ms = float(np.mean(generation_times) * 1000) if generation_times else 0.0
        fidelity_mean = float(np.mean(fidelity_scores)) if fidelity_scores else None
        stability_mean = float(np.mean(stability_scores)) if stability_scores else None
        self.logger.info(
            f"Generated {samples_done} {cam_method} explanations "
            f"(avg {avg_time_ms:.2f} ms/sample)."
        )
        return {
            'method': xai_cfg.get('method', 'gradcam'),
            'variant': cam_method,
            'samples': samples_done,
            'mc_samples': mc_samples,
            'avg_time_ms': avg_time_ms,
            'fidelity': fidelity_mean,
            'stability': stability_mean
        }

    def _generate_saliency(self, num_samples, xai_cfg):
        try:
            from explainability.saliency import SaliencyMap
        except ImportError:
            self.logger.warning("SaliencyMap not found, skipping explanations.")
            return {}

        self.logger.info("Generating Saliency explanations...")

        compute_xai_metrics = bool(xai_cfg.get('compute_metrics', True))
        saliency_gen = SaliencyMap(self.model)

        save_path = os.path.join(self.save_dir, 'xai')
        os.makedirs(save_path, exist_ok=True)

        samples_done = 0
        generation_times = []
        fidelity_scores = []
        stability_scores = []
        self.model.eval()

        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            for i in range(inputs.size(0)):
                if samples_done >= num_samples:
                    break

                single_input = inputs[i:i+1]
                true_label = labels[i].item()

                with torch.no_grad():
                    outputs = self.model(single_input)
                    probs = torch.softmax(outputs, dim=1)
                    conf, predicted_label = probs.max(1)
                    conf = conf.item()
                    predicted_label = predicted_label.item()

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                heatmap = saliency_gen.generate(single_input, target_class=predicted_label)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                generation_times.append(time.time() - start_time)

                if compute_xai_metrics:
                    fidelity_scores.append(
                        self._calculate_fidelity(single_input, predicted_label, heatmap, conf)
                    )
                    noisy_heatmap = saliency_gen.generate(
                        self._noisy_input(single_input),
                        target_class=predicted_label
                    )
                    stability_scores.append(self._calculate_stability(heatmap, noisy_heatmap))

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

                img_tensor = single_input[0] * std + mean
                img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)

                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = heatmap_color * 0.4 + img_np
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                class_name = self.classes.get(true_label, f"Class {true_label}")
                plt.title(f"Original: {class_name}")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                pred_name = self.classes.get(predicted_label, f"Class {predicted_label}")
                color = 'green' if predicted_label == true_label else 'red'
                plt.text(0.5, 0.6, f"Pred: {pred_name}", fontsize=12, ha='center', color=color)
                plt.text(0.5, 0.4, f"Conf: {conf:.4f}", fontsize=12, ha='center')
                plt.title("Prediction Details")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title("Saliency")
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'saliency_sample_{samples_done}.png'))
                plt.close()

                samples_done += 1

            if samples_done >= num_samples:
                break

        avg_time_ms = float(np.mean(generation_times) * 1000) if generation_times else 0.0
        fidelity_mean = float(np.mean(fidelity_scores)) if fidelity_scores else None
        stability_mean = float(np.mean(stability_scores)) if stability_scores else None
        self.logger.info(
            f"Generated {samples_done} Saliency explanations "
            f"(avg {avg_time_ms:.2f} ms/sample)."
        )
        return {
            'method': 'saliency',
            'variant': 'saliency',
            'samples': samples_done,
            'avg_time_ms': avg_time_ms,
            'fidelity': fidelity_mean,
            'stability': stability_mean
        }

    def _generate_integrated_gradients(self, num_samples, xai_cfg):
        try:
            from explainability.integrated_gradients import IntegratedGradients
        except ImportError:
            self.logger.warning("Integrated Gradients not found, skipping explanations.")
            return {}

        self.logger.info("Generating Integrated Gradients explanations...")

        steps = int(xai_cfg.get('steps', xai_cfg.get('ig_steps', 50)))
        compute_xai_metrics = bool(xai_cfg.get('compute_metrics', True))
        ig_gen = IntegratedGradients(self.model, steps=steps)

        save_path = os.path.join(self.save_dir, 'xai')
        os.makedirs(save_path, exist_ok=True)

        samples_done = 0
        generation_times = []
        fidelity_scores = []
        stability_scores = []
        self.model.eval()

        for inputs, labels in self.dataloader:
            inputs = inputs.to(self.device)
            for i in range(inputs.size(0)):
                if samples_done >= num_samples:
                    break

                single_input = inputs[i:i+1]
                true_label = labels[i].item()

                with torch.no_grad():
                    outputs = self.model(single_input)
                    probs = torch.softmax(outputs, dim=1)
                    conf, predicted_label = probs.max(1)
                    conf = conf.item()
                    predicted_label = predicted_label.item()

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                heatmap = ig_gen.generate(single_input, target_class=predicted_label)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                generation_times.append(time.time() - start_time)

                if compute_xai_metrics:
                    fidelity_scores.append(
                        self._calculate_fidelity(single_input, predicted_label, heatmap, conf)
                    )
                    noisy_heatmap = ig_gen.generate(
                        self._noisy_input(single_input),
                        target_class=predicted_label
                    )
                    stability_scores.append(self._calculate_stability(heatmap, noisy_heatmap))

                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

                img_tensor = single_input[0] * std + mean
                img_np = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                img_np = (img_np * 255).astype(np.uint8)

                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                overlay = heatmap_color * 0.4 + img_np
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)

                plt.figure(figsize=(15, 5))

                plt.subplot(1, 3, 1)
                plt.imshow(img_np)
                class_name = self.classes.get(true_label, f"Class {true_label}")
                plt.title(f"Original: {class_name}")
                plt.axis('off')

                plt.subplot(1, 3, 2)
                pred_name = self.classes.get(predicted_label, f"Class {predicted_label}")
                color = 'green' if predicted_label == true_label else 'red'
                plt.text(0.5, 0.6, f"Pred: {pred_name}", fontsize=12, ha='center', color=color)
                plt.text(0.5, 0.4, f"Conf: {conf:.4f}", fontsize=12, ha='center')
                plt.title("Prediction Details")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(overlay)
                plt.title("Integrated Gradients")
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'integrated_gradients_sample_{samples_done}.png'))
                plt.close()

                samples_done += 1

            if samples_done >= num_samples:
                break

        avg_time_ms = float(np.mean(generation_times) * 1000) if generation_times else 0.0
        fidelity_mean = float(np.mean(fidelity_scores)) if fidelity_scores else None
        stability_mean = float(np.mean(stability_scores)) if stability_scores else None
        self.logger.info(
            f"Generated {samples_done} Integrated Gradients explanations "
            f"(avg {avg_time_ms:.2f} ms/sample)."
        )
        return {
            'method': 'integrated_gradients',
            'variant': 'integrated_gradients',
            'samples': samples_done,
            'steps': steps,
            'avg_time_ms': avg_time_ms,
            'fidelity': fidelity_mean,
            'stability': stability_mean
        }
