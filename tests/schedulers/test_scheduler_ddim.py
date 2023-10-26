import unittest
import torch
import numpy as np

from src.schedulers.scheduler_ddim import DDIMScheduler

class TestDDIMScheduler(unittest.TestCase):

    def test_set_timesteps(self):
        scheduler = DDIMScheduler()
        device = torch.device('cpu')
        num_inference_steps = 10
        
        target_steps = 0
        expected_timesteps = torch.from_numpy(
            np.array([999, 899, 799, 699, 599, 499, 399, 299, 199, 99]) + scheduler.config.steps_offset)
        scheduler.set_timesteps(num_inference_steps, target_steps, device)
        assert scheduler.num_inference_steps == num_inference_steps
        assert scheduler.target_steps == target_steps
        assert torch.equal(scheduler.timesteps, expected_timesteps)
        target_steps = 10
        expected_timesteps = torch.from_numpy(
            np.array([999, 900, 801, 702, 603, 504, 405, 306, 207, 108]) + scheduler.config.steps_offset)
        scheduler.set_timesteps(num_inference_steps, target_steps, device)
        assert scheduler.num_inference_steps == num_inference_steps
        assert scheduler.target_steps == target_steps
        assert torch.equal(scheduler.timesteps, expected_timesteps)
        
        target_steps = np.asarray([0, 10])
        expected_timesteps = torch.from_numpy(np.vstack([
            (np.array([999, 899, 799, 699, 599, 499, 399, 299, 199, 99]) + scheduler.config.steps_offset),
            (np.array([999, 900, 801, 702, 603, 504, 405, 306, 207, 108]) + scheduler.config.steps_offset),
        ])).T
        scheduler.set_timesteps(num_inference_steps, target_steps, device)
        assert scheduler.num_inference_steps == num_inference_steps
        assert np.equal(scheduler.target_steps, target_steps).all()
        assert torch.equal(scheduler.timesteps, expected_timesteps)

        