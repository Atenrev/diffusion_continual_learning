import torch
import unittest

from src.continual_learning.metrics.diffusion_metrics import FIDMetric
from src.continual_learning.metrics.diffusion_metrics import DistributionMetrics


class TestFIDMetric(unittest.TestCase):
    def test_fid_metric(self):
        fid_metric = FIDMetric(device="cpu")
        true_y = torch.randn(10, 3, 32, 32)
        predicted_y = torch.randn(10, 3, 32, 32)

        fid_metric.update_true(true_y)
        fid_metric.update_predicted(predicted_y)

        fid_score = fid_metric.result()
        self.assertIsInstance(fid_score, float)


class TestDistributionMetrics(unittest.TestCase):
    def test_distribution_metrics(self):
        distribution_metrics = DistributionMetrics()
        true_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predicted_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        distribution_metrics.update_true(true_y)
        distribution_metrics.update_predicted(predicted_y)

        ard, kl_div, ratio_true, ratio_pred = distribution_metrics.result()
        self.assertIsInstance(ard, float)
        self.assertIsInstance(kl_div, float)
        self.assertIsInstance(ratio_true, torch.Tensor)
        self.assertIsInstance(ratio_pred, torch.Tensor)
        self.assertEqual(ratio_true.shape, (10,))
        self.assertEqual(ratio_pred.shape, (10,))
        self.assertAlmostEqual(ard, 0.0, delta=1e-6)
        self.assertAlmostEqual(kl_div, 0.0, delta=1e-6)

    def test_distribution_metrics_with_different_distributions(self):
        distribution_metrics = DistributionMetrics()
        true_y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        predicted_y = torch.tensor([0, 1, 1, 1, 4, 5, 6, 7, 8, 8])

        distribution_metrics.update_true(true_y)
        distribution_metrics.update_predicted(predicted_y)

        ard, kl_div, ratio_true, ratio_pred = distribution_metrics.result()
        self.assertIsInstance(ard, float)
        self.assertIsInstance(kl_div, float)
        self.assertIsInstance(ratio_true, torch.Tensor)
        self.assertIsInstance(ratio_pred, torch.Tensor)
        self.assertEqual(ratio_true.shape, (10,))
        self.assertEqual(ratio_pred.shape, (10,))
        self.assertGreater(ard, 0.0)
        self.assertGreater(kl_div, 0.0)