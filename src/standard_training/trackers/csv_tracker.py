import os
import json

from src.standard_training.trackers.base_tracker import Stage, ExperimentTracker


class CSVTracker(ExperimentTracker):
    """
    Creates a tracker that implements the ExperimentTracker protocol and logs to a CSV file.
    """

    def __init__(self, configs: dict, results_path: str):
        self.stage = Stage.TRAIN
        self.csv_file_train = open(os.path.join(results_path, "train.csv"), "w")
        self.csv_file_test = open(os.path.join(results_path, "test.csv"), "w")

        # Save configs
        with open(os.path.join(results_path, "configs.json"), "w") as f:
            json.dump(configs, f)

        # Create CSV files
        self.csv_file_train.write("step,metric,value\n")
        self.csv_file_test.write("epoch,metric,value\n")

    def set_stage(self, stage: Stage):
        self.stage = stage

    def add_batch_metric(self, name: str, value: float, step: int, commit: bool = True):
        if self.stage == Stage.TRAIN:
            self.csv_file_train.write(f"{step},{name},{value}\n")
        else:
            pass

    def add_epoch_metric(self, name: str, value: float, step: int, commit: bool = True):
        if self.stage == Stage.TRAIN:
            pass
        else:
            self.csv_file_test.write(f"{step},{name},{value}\n")
        
    def flush(self):
        self.csv_file_train.flush()
        self.csv_file_test.flush()

    def finish(self):
        self.csv_file_train.close()
        self.csv_file_test.close()