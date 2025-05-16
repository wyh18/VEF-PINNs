from math import pi
import numpy as np
import torch
import math


class LinearTransport(object):
    """A class for solving moment-based transport equations using PINNs.

    Attributes:
        device (torch.device): Computation device (CPU/GPU)
        kn (float): Knudsen number
        sigma_s (float): Scattering coefficient (hardcoded for testing)
        sigma_a (float): Absorption coefficient (hardcoded for testing)
        domain (dict): Time and space domain boundaries
        vquads (torch.Tensor): Velocity quadrature points
        wquads (torch.Tensor): Quadrature weights
        ref_x (torch.Tensor): Reference spatial grid
        ref_t (torch.Tensor): Reference temporal grid
    """

    def __init__(self, config, sol, name="Linear_Transport_Eqn", **kwargs):
        """Initialize physical parameters and computational domain."""
        # Device configuration
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        )

        # Hardcoded parameters for testing
        self.kn = config["physical_config"]["kn"]
        self.sigma_s = 1  # Temporary override for testing
        self.sigma_a = 0  # Temporary override for testing
        self.sigma = self.kn ** 2 * self.sigma_a + self.sigma_s


        # Domain configuration
        self.tmin, self.tmax = config["physical_config"]["t_range"]
        self.xmin, self.xmax = config["physical_config"]["x_range"]
        self.vmin, self.vmax = config["physical_config"]["v_range"]

        # Gauss-Legendre quadrature setup
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

        self.ref_x.requires_grad_(True)
        self.ref_t.requires_grad_(True)

        # Boundary and initial conditions
        self.f_l = 2.0  # Left boundary value
        self.f_r = 1.0  # Right boundary value
        self.f_init = 1.0  # Initial condition constant

    def residual(self, sol, model_I, inputs):
        """Calculate PDE residual for the moment-based transport equation.

        Args:
            sol: Density network
            model_I: Moment network
            inputs: Tuple of (t, x, v) tensors

        Returns:
            Dictionary containing equation residuals
        """
        t, x, v = inputs
        values, derivatives = self.value_and_grad(sol, model_I, t, x, v)
        rho = values["rho"]
        # Extract derivatives
        rho_t, rho_x, rho_xx = derivatives["rho_grid"]
        aver_vf_x_t, eddington, edd_x, edd_xx = derivatives["f_grid"]

        # Assemble moment equation residual
        moment_res = rho_t - (
                self.kn * aver_vf_x_t
                + eddington * rho_xx
                + 2 * edd_x * rho_x
                + edd_xx * rho
        )/self.sigma

        return {"equation": moment_res}

    def value_and_grad(self, sol, model_I, t, x, v):
        """Compute solution values and higher-order derivatives."""
        t.requires_grad_(True)
        x.requires_grad_(True)

        # Density network evaluation
        rho = sol(torch.cat([t, x], -1))
        values = {"rho": rho}

        # First and second derivatives of density
        drho_dt = self._compute_gradient(rho, t)
        drho_dx = self._compute_gradient(rho, x)
        drho_dxx = self._compute_gradient(drho_dx, x)

        # Moment calculations
        aver_f = self.average_op(model_I, [t, x], [self.vquads, self.wquads])
        aver_vf = self.average_op(model_I, [t, x],
                                  [self.vquads, self.wquads * self.vquads])
        aver_vvf = self.average_op(model_I, [t, x],
                                   [self.vquads, self.wquads * self.vquads ** 2])

        # Eddington tensor calculations
        eddington = aver_vvf / (aver_f + 1e-15)
        edd_x = self._compute_gradient(eddington, x)
        edd_xx = self._compute_gradient(edd_x, x)

        # Velocity moment derivatives
        aver_vf_x = self._compute_gradient(aver_vf, x)
        aver_vf_xt = self._compute_gradient(aver_vf_x, t)
        return values, {
            "rho_grid": (drho_dt, drho_dx, drho_dxx),
            "f_grid": (aver_vf_xt, eddington, edd_x, edd_xx)
        }

    def _compute_gradient(self, output, input):
        """Helper function for gradient calculation."""
        return torch.autograd.grad(
            outputs=output,
            inputs=input,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

    def average_op(self, model, t_x, vwquads):
        """Compute velocity-space average using quadrature.

        Args:
            model: Neural network to evaluate
            t_x: List of (t, x) tensors
            vwquads: Tuple of (velocity points, weights)

        Returns:
            torch.Tensor: Integrated moment value
        """
        t, x = t_x
        v, w = vwquads

        # Expand inputs to quadrature points
        tx = torch.cat([t, x], -1)[:, None, :]  # Shape: [N, 1, 2]

        # expanded_tx = tx.expand(-1, len(v), -1)
        expanded_tx = tx * torch.ones((tx.shape[0], len(v), 1),
                                      device=self.device)
        expanded_v = v[None, :, None].expand(tx.shape[0], -1, -1)

        # Evaluate model and integrate
        In = model(torch.cat([expanded_tx, expanded_v], -1))
        return 0.5 * torch.sum(In * w[None, :, None], dim=1)

    def bc(self, sol, model_I, inputs):
        """Periodic boundary condition implementation."""
        tbc, vbc = inputs

        # Boundary evaluations
        rho_left = sol(torch.cat([tbc, self.xmin * torch.ones_like(tbc)], -1))
        rho_right = sol(torch.cat([tbc, self.xmax * torch.ones_like(tbc)], -1))

        return {"rho_bc": (rho_left - rho_right, rho_right - rho_left)}

    def ic(self, sol, inputs):
        """Initial condition implementation."""
        xic, _ = inputs
        rho_pred = sol(torch.cat([self.tmin * torch.ones_like(xic), xic], -1))

        # Analytical initial condition (cosine perturbation)
        rho_analytical = (1 + torch.cos(4 * pi * xic)) / math.sqrt(2 * pi) * 0.922562 + 1
        return {"initial": rho_pred - rho_analytical}

    def val(self, sol, ref):
        """Validation metric: Relative L2 error for density."""
        density_pred = self.rho(sol, [self.ref_t, self.ref_x]).cpu()
        density_ref = ref.cpu()

        # Relative L2 error with numerical stability
        mse = torch.mean((density_ref - density_pred) ** 2)
        norm = torch.mean(density_ref ** 2)
        return torch.sqrt(mse / (norm + 1e-8))

    def rho(self, sol, inputs):
        """Direct density evaluation helper."""
        t, x = inputs
        return sol(torch.cat([t, x], -1))