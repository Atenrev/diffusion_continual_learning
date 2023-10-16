import os
import csv
import shutil
import unittest

from unittest.mock import MagicMock

from src.continual_learning.loggers import CSVLogger


class TestCSVLogger(unittest.TestCase):
    def setUp(self):
        self.logger = CSVLogger(log_folder="test_logs")

    def tearDown(self):
        self.logger.close()
        shutil.rmtree("test_logs")

    def test_init(self):
        self.assertTrue(os.path.exists("test_logs"))
        self.assertTrue(os.path.exists("test_logs/training_results.csv"))
        self.assertTrue(os.path.exists("test_logs/eval_results.csv"))

    def test_print_csv_headers(self):
        with open("test_logs/training_results.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["metric_name", "training_exp", "epoch", "x_plot", "value"])

        with open("test_logs/eval_results.csv", "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            self.assertEqual(header, ["metric_name", "eval_exp", "training_exp", "value"])

    def test_log_single_metric(self):
        self.logger.metric_vals = {}
        self.logger.log_single_metric("accuracy", 0.9, 1)
        self.logger.log_single_metric("loss", 0.1, 1)
        self.assertEqual(self.logger.metric_vals, {"accuracy": [("accuracy", 1, 0.9)], "loss": [("loss", 1, 0.1)]})

    def test_print_train_metrics(self):
        self.logger.training_file = MagicMock()
        self.logger.metric_vals = {"accuracy": [("accuracy", 1, 0.9)], "loss": [("loss", 1, 0.1)]}
        self.logger.print_train_metrics(1, 1)
        args = self.logger.training_file.write.call_args_list
        self.assertEqual("".join([arg[0][0] for arg in args]), "accuracy,1,1,1,0.9000\nloss,1,1,1,0.1000\n")

    def test_print_eval_metrics(self):
        self.logger.eval_file = MagicMock()
        self.logger.metric_vals = {"accuracy": [("accuracy", 1, 0.9)], "loss": [("loss", 1, 0.1)]}
        self.logger.print_eval_metrics(1, 2)
        # assert it's called many times to form "accuracy,1,2,0.9\nloss,1,2,0.1\n"
        args = self.logger.eval_file.write.call_args_list
        self.assertEqual("".join([arg[0][0] for arg in args]), "accuracy,1,2,0.9000\nloss,1,2,0.1000\n")
