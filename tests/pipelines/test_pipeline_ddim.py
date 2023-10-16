import torch
import unittest

from src.schedulers.scheduler_ddim import DDIMScheduler
from src.pipelines.pipeline_ddim import DDIMPipeline

class TestDDIMPipeline(unittest.TestCase):

    def dummy_model(self):
        class DummyModel:
            def __init__(self) -> None:
                config = unittest.mock.Mock()
                config.in_channels = torch.tensor(1)
                config.sample_size = torch.tensor((32, 32))
                self.config = config
                self.dtype = torch.float32
                
            def __call__(self, sample, t, *args):
                mock_response = unittest.mock.Mock()
                mock_response.sample = sample.clone()

                for s in range(sample.shape[0]):
                    for i in range(sample.shape[2]):
                        if mock_response.sample[s, 0, 0, i] >= t[s]:
                            continue
                        mock_response.sample[s, 0, 0, i] = t[s]
                        break

                return mock_response
            
        model = DummyModel()
        return model

    def setUp(self):
        self.unet = self.dummy_model()
        self.scheduler = DDIMScheduler()
        self.pipeline = DDIMPipeline(self.unet, self.scheduler)

    def test_mask(self):
        batch_size = 2
        generator = torch.Generator()
        generator.manual_seed(42)
        target_timesteps = torch.tensor([0, 999])
        output = self.pipeline(batch_size=batch_size, num_inference_steps=3, 
                               target_steps=target_timesteps, use_clipped_model_output=None, 
                               output_type="torch_raw", return_dict=False, generator=generator)
        self.assertEqual(output.shape, (batch_size, 1, 32, 32))
        unmasked_output = output[0, 0, 0]
        masked_output = output[1, 0, 0]
        self.assertGreater(masked_output[0], 2)
        self.assertGreater(masked_output[1], 2)
        self.assertLess(masked_output[2], 2)
        self.assertEqual(abs(unmasked_output[0]), 1)
        self.assertEqual(abs(unmasked_output[1]), 1)
        self.assertEqual(abs(unmasked_output[2]), 1)