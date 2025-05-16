from math import pi
import numpy as np
import torch
import math


class LinearTransport(object):
    """A class for solving the linear transport equation using PINNs (Physics-Informed Neural Networks).

    Attributes:
        device (torch.device): Computation device (CPU/GPU)
        kn (float): Knudsen number
        sigma_s (float): Scattering cross-section
        sigma_a (float): Absorption cross-section
        sigma (torch.Tensor): Total cross-section
        domain (dict): Time and space domain boundaries
        vquads (torch.Tensor): Velocity quadrature points
        wquads (torch.Tensor): Quadrature weights
        ref_x (torch.Tensor): Reference spatial grid
        ref_t (torch.Tensor): Reference temporal grid
        I_l (float): Left boundary value
        I_r (float): Right boundary value
        I_init (float): Initial condition value
    """

    def __init__(self, config, sol, name="Linear_Transport_Eqn", **kwargs):
        """Initialize physical parameters and computational domain."""
        # Device configuration
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(
                device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

        # Physical parameters
        self.kn = config["physical_config"]["kn"]
        self.sigma_s = config["physical_config"]["sigma_s"]
        self.sigma_a = config["physical_config"]["sigma_a"]
        self.sigma = self.kn ** 2 * self.sigma_a + self.sigma_s

        # Domain configuration
        self.tmin, self.tmax = config["physical_config"]["t_range"]
        self.xmin, self.xmax = config["physical_config"]["x_range"]
        self.vmin, self.vmax = config["physical_config"]["v_range"]

        # Velocity quadrature setup (Gauss-Legendre)
        self.num_vquads = config["model_config"]["num_vquads"]
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        vquads = 0.5 * (vquads + 1.0) * (self.vmax - self.vmin) + self.vmin
        wquads = 0.5 * (self.vmax - self.vmin) * wquads
        self.vquads = torch.Tensor(vquads).to(self.device)
        self.wquads = torch.Tensor(wquads).to(self.device)

        # Reference grid for validation
        self.nx = config["physical_config"]["ref_N"]
        self.ref_x_min = config["physical_config"]["ref_x_range"][0]
        self.ref_x_max = config["physical_config"]["ref_x_range"][1]

        self.ref_x = torch.Tensor(np.linspace(self.ref_x_min, self.ref_x_max, self.nx).reshape((self.nx, 1))).to(
            self.device)
        self.ref_t = self.tmax * torch.ones((self.nx, 1)).to(self.device)

        self.ref_x.requires_grad = True
        self.ref_t.requires_grad = True

        # Boundary and initial conditions
        self.I_l = 2.0  # Left boundary value
        self.I_r = 1.0  # Right boundary value
        self.I_init = 1.0  # Initial condition constant

    def exact_I(self, t, x, v):
        I = torch.sin(2 * torch.pi * (x + t)) / (2 * torch.pi) - self.kn * v ** 2 * torch.cos(
            2 * torch.pi * (x + t)) + 2
        return I

    def exact_rho(self, t, x):
        rho = torch.sin(2 * torch.pi * (x + t)) / (2 * torch.pi) + 2 - 1 / 3 * self.kn * torch.cos(
            2 * torch.pi * (x + t))
        return rho

    # inputs: (t, x, v)

    def exact_eps2_source(self, t, x, v):
        eps2Q = ((self.kn + v + self.sigma_s / 3 - v ** 2 * self.sigma_s - self.sigma_a * self.kn ** 2 * v ** 2)
                 * torch.cos(2 * torch.pi * (x + t)) + self.kn * (
                             2 * torch.pi * self.kn * v ** 2 + 2 * torch.pi * v ** 3 + self.sigma_a / (
                                 2 * torch.pi)) * torch.sin(2 * torch.pi * (x + t)) + self.kn * 2 * self.sigma_a)
        return eps2Q

    def residual(self, sol, model_rho, inputs):
        """Calculate PDE residual for the linear transport equation.

        Args:
            sol: Neural network solution approximator
            model_rho: Density estimator network
            inputs: Tuple of (t, x, v) tensors

        Returns:
            Dictionary containing equation residuals
        """
        t, x, v = inputs
        values, derivatives = self.value_and_grad(sol, t, x, v)
        I = values["I"]
        I_t, I_x = derivatives["I"]
        rho = model_rho(torch.cat([t, x], -1))

        # The source: \varepsilon^2 * Q
        eps2Q = self.exact_eps2_source(t, x, v)

        # Transport equation residual
        transport_res = (
                self.kn * I_t
                + v * I_x
                - ((rho - I) * self.sigma_s / self.kn
                   - self.kn * self.sigma_a * I
                   + eps2Q)
        )

        return {"equation": transport_res}

    def value_and_grad(self, sol, t, x, v):
        """Compute solution values and gradients.

        Returns:
            tuple: (values dict, gradients dict)
        """
        t.requires_grad_(True)
        x.requires_grad_(True)

        # Forward pass
        I = sol(torch.cat([t, x, v], -1))

        # Compute gradients using automatic differentiation
        dI_dt = torch.autograd.grad(
            I, t,
            grad_outputs=torch.ones_like(I),
            create_graph=True,
            retain_graph=True
        )[0]

        dI_dx = torch.autograd.grad(
            I, x,
            grad_outputs=torch.ones_like(I),
            create_graph=True,
            retain_graph=True
        )[0]

        return {"I": I}, {"I": (dI_dt, dI_dx)}

    def rho(self, sol, inputs):
        """Compute density by integrating over velocity space."""
        t, x = inputs
        tx = torch.cat([t, x], -1)[:, None, :]  # Shape: [N, 1, 2]

        # Expand to quadrature points
        expanded_tx = tx * torch.ones((tx.shape[0], len(self.vquads), 1),
                                      device=self.device)
        expanded_v = self.vquads[None, :, None].expand(tx.shape[0], -1, -1)

        # Evaluate neural solution
        In = sol(torch.cat([expanded_tx, expanded_v], -1))

        # Numerical integration using quadrature weights
        return torch.sum(In * self.wquads[None, :, None], dim=-2) * 0.5

    def bc(self, sol, inputs):
        """In-flow boundary condition implementation."""
        tbc, vbc = inputs
        vbc_l, vbc_r = vbc, -vbc
        # Left boundary evaluation
        Ibc_l = sol(torch.cat([
            tbc,
            self.xmin * torch.ones_like(tbc),
            vbc_l
        ], -1))

        # Right boundary evaluation
        Ibc_r = sol(torch.cat([
            tbc,
            self.xmax * torch.ones_like(tbc),
            vbc_r
        ], -1))
        res_l = Ibc_l - self.exact_I(tbc, self.xmin * torch.ones_like(tbc), vbc_l)
        res_r = Ibc_l - self.exact_I(tbc, self.xmin * torch.ones_like(tbc), vbc_l)
        return {"Ibc": (res_l, res_r)}

    def ic(self, sol, inputs):
        """Initial condition implementation."""
        xic, vic = inputs

        # Neural network prediction at t=0
        I0 = sol(torch.cat([
            self.tmin * torch.ones_like(xic),
            xic,
            vic
        ], -1))

        # Analytical initial condition

        res_ic = I0 - self.exact_I(self.tmin * torch.ones_like(xic), xic, vic)

        return {"initial": res_ic}

    def val(self, sol, ref):
        """Validation metric: Relative L2 error for density."""
        density_pred = self.rho(sol, [self.ref_t, self.ref_x]).cpu()
        density_ref = self.exact_rho(self.ref_t, self.ref_x).cpu()

        # Relative L2 error calculation
        mse = torch.mean((density_ref - density_pred) ** 2)
        norm = torch.mean(density_ref ** 2)
        return torch.sqrt(mse / (norm + 1e-8))
