import torch

class Sampler(object):
    def __init__(self, config, name="sampler", **kwargs):
        dataset_config = config["dataset_config"]
        self.interior_samples = dataset_config["interior_samples"]
        self.boundary_samples = dataset_config["boundary_samples"]
        self.initial_samples = dataset_config["initial_samples"]

        physical_config = config["physical_config"]
        self.t_range = physical_config["t_range"]
        self.x_range = physical_config["x_range"]
        self.v_range = physical_config["v_range"]

        device_ids = config["model_config"]["device_ids"]
        primary_device = device_ids[0] if device_ids else 0
        self.device = torch.device(
            f"cuda:{primary_device}"
            if torch.cuda.is_available() else "cpu"
        )

    def _generate_uniform(self, shape, low, high):
        """Generate uniformly distributed tensor with specified shape and range"""
        return low + torch.rand(shape, device=self.device) * (high - low)

    def interior(self):
        """Sample interior points: returns (t, x, v)"""
        t = self._generate_uniform((self.interior_samples, 1), 0, self.t_range[-1])
        x = self._generate_uniform((self.interior_samples, 1), *self.x_range)
        v = self._generate_uniform((self.interior_samples, 1), *self.v_range)
        return t, x, v

    def boundary(self):
        """Sample inflow boundary conditions: returns (t, v)"""
        t = self._generate_uniform((self.boundary_samples, 1), 0, self.t_range[-1])
        v = self._generate_uniform((self.boundary_samples, 1), 0, self.v_range[-1])
        return t, v

    def initial(self):
        """Sample initial conditions: returns (x, v)"""
        x = self._generate_uniform((self.initial_samples, 1), *self.x_range)
        v = self._generate_uniform((self.initial_samples, 1), *self.v_range)
        return x, v