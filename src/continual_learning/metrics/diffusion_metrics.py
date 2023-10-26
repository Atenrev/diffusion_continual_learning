import os
import torch

import torch.nn as nn
from torchvision.models import resnet18
from torchmetrics.image.fid import FrechetInceptionDistance
from avalanche.training.templates import SupervisedTemplate
from avalanche.evaluation import Metric, PluginMetric
from avalanche.evaluation.metric_results import MetricResult, MetricValue
from avalanche.evaluation.metric_utils import get_metric_name


class FIDMetric(Metric[float]):
    """
    This metric computes the Frechet Inception Distance (FID) between two
    distributions of images. It uses the FID implementation from
    `torchmetrics <https://torchmetrics.readthedocs.io/en/latest/>`.
    """
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()

    @torch.no_grad()
    def update_true(
        self,
        true_y: torch.Tensor,
    ) -> None:
        true_y = torch.as_tensor(true_y)

        if true_y.min() < 0:
            true_y = (true_y + 1) / 2

        if true_y.shape[1] == 1:
            true_y = torch.cat([true_y] * 3, dim=1)

        self.fid.update(true_y, real=True)

    @torch.no_grad()
    def update_predicted(
        self,
        predicted_y: torch.Tensor,
    ) -> None:
        predicted_y = torch.as_tensor(predicted_y)

        if len(predicted_y) == 3: 
            # Not expected from a dm output
            predicted_y = predicted_y[0]

        if predicted_y.shape[1] == 1:
            predicted_y = torch.cat([predicted_y] * 3, dim=1)

        self.fid.update(predicted_y, real=False)

    def result(self) -> float:
        return self.fid.compute().cpu().detach().item()

    def reset(self):
        self.fid = FrechetInceptionDistance(normalize=True, feature=2048)
        if type(self.device) == str and self.device.startswith('cuda') \
                or type(self.device) == torch.device and self.device.type == 'cuda':
            self.fid.cuda()


class DistributionMetrics(Metric[float]):
    """
    This metric computes the Absolute Ratio Difference (ARD) and the KLD
    between two distributions of class frequencies. 

    ARD^{b} = \sum_{j=1}^b \lvert \rho_j^{T_b} - \rho_j^{\epsilon_b} \rvert
    """
    def __init__(self):
        self.reset()  

    @torch.no_grad()
    def update_true(
        self,
        true_y: torch.Tensor,
    ) -> None:
        true_y = torch.as_tensor(true_y)
        hist_true = torch.zeros(10)
        for i in range(10):
            hist_true[i] = torch.sum(true_y == i)
        self.hist_true += hist_true
        
    @torch.no_grad()
    def update_predicted(
        self,
        predicted_y: torch.Tensor,
    ) -> None:
        predicted_y = torch.as_tensor(predicted_y)
        hist_pred = torch.zeros(10)
        for i in range(10):
            hist_pred[i] = torch.sum(predicted_y == i)
        self.hist_pred += hist_pred

    def result(self) -> float:
        ratio_true = self.hist_true / torch.sum(self.hist_true)
        ratio_pred = self.hist_pred / torch.sum(self.hist_pred)

        ard = torch.sum(torch.abs(ratio_true - ratio_pred)) / 2.0
        kl = torch.nn.KLDivLoss(reduction='batchmean')
        kl_div = kl((ratio_pred + 1e-8).log()[None, :], (ratio_true + 1e-8)[None, :])
        return ard.cpu().detach().item(), kl_div.cpu().detach().item(), ratio_true, ratio_pred

    def reset(self):
        self.hist_true = torch.zeros(10)
        self.hist_pred = torch.zeros(10)


class DiffusionMetricsMetric(PluginMetric[float]):

    def __init__(self, device='cuda', weights_path: str = "results/cnn_fmnist/", n_samples: int = 10000, num_classes: int = 10):
        self.classifier = resnet18()
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
        self.classifier.to(device)
        self.classifier.load_state_dict(torch.load(os.path.join(weights_path, "resnet.pth"), map_location=device))
        self.classifier.eval() 

        self.fid_metric = FIDMetric(device)
        self.dist_metrics = DistributionMetrics()

        self.n_samples = n_samples

    def result(self) -> float:
        return self.fid_metric.result()

    def reset(self):
        self.fid_metric.reset()
        self.dist_metrics.reset()

    def before_training_exp(self, strategy: "SupervisedTemplate") -> None:
        super().before_training_exp(strategy)
        self.train_exp_id = strategy.experience.current_experience

    def after_training_exp(self, strategy: SupervisedTemplate) -> MetricResult:
        self.reset()

        batch_size = strategy.eval_mb_size
        num_samples = 10000
        num_batches = num_samples // batch_size
        remaining_samples = num_samples % batch_size

        for _ in range(num_batches):
            predicted_samples = strategy.generate_samples(batch_size)
            if predicted_samples.shape[1] == 1:
                predicted_samples = torch.cat([predicted_samples] * 3, dim=1)
            with torch.no_grad():
                classes = torch.argmax(self.classifier((predicted_samples - 0.5) * 2), dim=1)
            classes_np = classes.cpu().numpy()
            self.fid_metric.update_predicted(predicted_samples)
            self.dist_metrics.update_predicted(classes_np)

        if remaining_samples > 0:
            predicted_samples = strategy.generate_samples(remaining_samples)
            if predicted_samples.shape[1] == 1:
                predicted_samples = torch.cat([predicted_samples] * 3, dim=1)
            with torch.no_grad():
                classes = torch.argmax(self.classifier((predicted_samples - 0.5) * 2), dim=1)
            classes_np = classes.cpu().numpy()
            self.fid_metric.update_predicted(predicted_samples)
            self.dist_metrics.update_predicted(classes_np)

        return super().after_training_exp(strategy)

    def after_eval_iteration(self, strategy: 'PluggableStrategy'):
        """
        Update the accuracy metric with the current
        predictions and targets
        """            
        super().after_eval_iteration(strategy)

        if strategy.experience.current_experience <= self.train_exp_id:        
            self.fid_metric.update_true(strategy.mb_x)
            self.dist_metrics.update_true(strategy.mb_y)

    def _package_result(self, strategy):
        """
        Package the result for logging
        """
        add_exp = False
        plot_x_position = strategy.clock.train_iterations
        metrics = []

        metric_value = self.fid_metric.result()
        metric_name = get_metric_name("stream_fid", strategy,
                                    add_experience=add_exp,
                                    add_task=True)
        metrics.append(MetricValue(self, metric_name, metric_value,
                                plot_x_position))
        
        ard_val, kl_val, hist_true, hist_pred = self.dist_metrics.result()
        ard_name = get_metric_name("stream_ard", strategy,
                                        add_experience=add_exp,
                                        add_task=True)
        kl_name = get_metric_name("stream_kld", strategy,
                                        add_experience=add_exp,
                                        add_task=True)
        hist_pred_name = get_metric_name("stream_hist_pred", strategy,
                                        add_experience=add_exp,
                                        add_task=True)
        hist_true_name = get_metric_name("stream_hist_true", strategy,
                                        add_experience=add_exp,
                                        add_task=True)

        metrics.append(MetricValue(self, ard_name, ard_val, plot_x_position))
        metrics.append(MetricValue(self, kl_name, kl_val, plot_x_position))
        metrics.append(MetricValue(self, hist_pred_name, hist_pred, plot_x_position))
        metrics.append(MetricValue(self, hist_true_name, hist_true, plot_x_position))
                                     
        return metrics

    def after_eval(self, strategy: 'PluggableStrategy'):      
        return self._package_result(strategy)
    
    def __str__(self):
        return "DiffusionMetrics" 