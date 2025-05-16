import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

path = os.getcwd()
sys.path.append(path)
sys.path.append(os.path.abspath(os.path.join(path, "../..")))
sys.path.append("..")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from make_dir import mkdir
from load_yaml import get_yaml

import net as solutions
import equation.I_eqn as equation_I
import equation.rho_eqn as equation_rho
from Dataset import Sampler

import matplotlib.pyplot as plt
import time


class VEF_PINN(object):
    """
    Physics-Informed Neural Network with Variable Eddington Factor Iteration for Linear Radiation Transfer Equations

    Attributes:
        Config (dict): Configuration parameters
        ref_rho (Tensor): Reference radiation density data
        model_I (nn.Module): Radiation intensity network
        model_rho (nn.Module): Material density network
        device (torch.device): Computation device (GPU/CPU)
    """

    def __init__(self, name="VEF_PINN", **kwargs):
        """Initialize PINN solver components"""
        # Load configuration and reference data
        self._load_config()
        self._load_reference_data()

        # Initialize model dimensions
        self._init_model_dims()

        # Build neural networks
        self.model_I = self._build_model("I")
        self.model_rho = self._build_model("rho")

        # Configure hardware
        self.device = self._setup_device()
        self._move_models_to_device()

        # Initialize validation grid
        self._init_reference_grid()

    def _load_config(self):
        """Load YAML configuration file"""
        current_path = os.path.abspath("")
        yaml_path = os.path.join(current_path, "Config.yaml")
        self.Config = get_yaml(yaml_path)

    def _load_reference_data(self):
        """Load reference validation data"""
        ref_path = os.path.join(os.path.abspath("../.."), "data/e2_ref_kn1e0_t=01.npz")
        ref_data = np.load(ref_path)
        self.ref_rho = torch.Tensor(ref_data["macro_frames"]).cpu().reshape(-1, 1)

    def _init_model_dims(self):
        """Initialize model input dimensions"""
        pc = self.Config["physical_config"]
        self.d_in = pc["time_dimension"] + pc["space_dimension"] + pc["velocity_dimension"]
        self.d_in_rho = pc["time_dimension"] + pc["space_dimension"]

    def _build_model(self, phys_type: str) -> nn.Module:
        """
        Construct neural network for specific physical quantity

        Args:
            phys_type: Physical quantity type ('I', 'rho')

        Returns:
            Initialized neural network
        """
        # Get network architecture
        layer_key = f"units_{phys_type}"
        layers = self.Config["model_config"][layer_key]
        input_size = getattr(self, f"d_in_{phys_type}" if phys_type != "I" else "d_in")

        # Dynamic model creation
        model_class = getattr(solutions, f"Model_{self.Config['model_config']['neural_network_type']}")
        model = model_class(input_size=input_size, layers=layers, output_size=1)

        # Parameter initialization
        solutions.Xavier_initi(model)
        return model

    def _setup_device(self) -> torch.device:
        """Configure computation device with multi-GPU support"""
        device_ids = self.Config["model_config"]["device_ids"]
        return torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    def _move_models_to_device(self):
        """Transfer models to target device"""
        self.model_I.to(self.device)
        self.model_rho.to(self.device)

    def _init_reference_grid(self):
        """Initialize validation spatial grid"""
        self.nx = self.Config["physical_config"]["ref_N"]
        self.ref_x_min = self.Config["physical_config"]["ref_x_range"][0]
        self.ref_x_max = self.Config["physical_config"]["ref_x_range"][1]

        self.ref_x = torch.Tensor(np.linspace(self.ref_x_min, self.ref_x_max, self.nx)
                                  .reshape((self.nx, 1))).to(self.device)
        self.ref_t = torch.full_like(self.ref_x, self.Config["physical_config"]["t_range"][-1])

    def _load_pretrained_models(self):
        """Load pretrained model parameters"""
        solutions.load_param(self.model_I, './model_saved/model_I_params.pkl')
        solutions.load_param(self.model_rho, './model_saved/model_rho_params.pkl')

    def _print_training_info(self, iteration, total_iter, loss, metrics, optimizer, model_type):
        """
        Print training progress information

        Args:
            iteration: Current iteration count
            total_iter: Total iterations
            loss: Current loss value
            metrics: Loss component metrics
            optimizer: Optimization object
            model_type: Model type ('I', 'rho')
        """
        lr = optimizer.param_groups[0]["lr"]
        base_info = f"[Iter: {iteration + 1:6d}/{total_iter:6d} | LR: {lr:.2e} | Loss: {loss.item():.2e}]"

        error_type = {
            "I": "Radiation",
            "rho": "Density",
        }[model_type]
        error_info = f"[{error_type} Error: {metrics['error']:.2e}]"

        loss_components = [
            f"Eqn: {metrics['res_eqn']:.2e}",
            f"BC: {metrics['res_bc']:.2e}",
            f"IC: {metrics['res_ic']:.2e}"
        ]

        full_info = f"{base_info}\n{error_info}\n[Loss Components: {', '.join(loss_components)}]"
        print(full_info)
        print("-" * 80)

    def _compute_loss(self, model, eqn, trainloader, model_type):
        """
        Calculate total loss and metrics

        Returns:
            tuple: (Total loss, metrics dictionary)
        """
        with torch.set_grad_enabled(True):
            if model_type == "I":
                eqn_residual = eqn.residual(sol=model, model_rho=self.model_rho, inputs=trainloader[0])
            elif model_type == "rho":
                eqn_residual = eqn.residual(sol=model, model_I=self.model_I, inputs=trainloader[0])

        res = eqn_residual["equation"]
        res_eqn = torch.mean(res ** 2)
        res_bc, res_ic = 0.0, 0.0

        # Boundary conditions
        if model_type in ["I", "rho"]:
            if model_type == "I":
                boundary_res = eqn.bc(sol=model, inputs=trainloader[1])
                res_I_left, res_I_right = boundary_res["Ibc"]
                res_bc = torch.mean(res_I_left ** 2) + torch.mean(res_I_right ** 2)
            else:
                boundary_res = eqn.bc(sol=model, model_I=self.model_I, inputs=trainloader[1])
                res_rho_l, res_rho_r = boundary_res["rho_bc"]
                res_bc = torch.mean(res_rho_l ** 2) + torch.mean(res_rho_r ** 2)

        # Initial conditions
        init_res = eqn.ic(sol=model, inputs=trainloader[2])
        res_ic = torch.mean(init_res["initial"] ** 2)

        # Loss composition
        regularizers = self.Config["model_config"]["regularizers"]
        loss_components = [
            regularizers[0] * res_eqn,
            regularizers[1] * res_bc,
            regularizers[-1] * res_ic
        ]
        total_loss = sum(loss_components)

        # Validation error
        with torch.no_grad():
            error = eqn.val(sol=model, ref=self.ref_rho)

        return total_loss, {
            "error": error.item(),
            "res_eqn": res_eqn.item(),
            "res_bc": res_bc.item(),
            "res_ic": res_ic.item()
        }

    def _train_loop(self, model, eqn_module, model_type: str):
        """
        Generic training loop

        Args:
            model: Target model to train
            eqn_module: Corresponding equation module
            model_type: Model type ('I', 'rho')
        """
        # Optimizer setup
        optimizer = optim.Adam(model.parameters(), lr=self.Config["model_config"]["lr"])
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=self.Config["model_config"]["stage_num"],
            gamma=self.Config["model_config"]["decay_rate"]
        )

        self._load_pretrained_models()
        eqn = eqn_module(config=self.Config, sol=model)
        mkdir("model_saved"), mkdir("record"), mkdir("figure")

        iterations = self.Config["model_config"][f"{model_type}_iterations"]
        for it in range(iterations):
            sampler = Sampler(self.Config)
            trainloader = [sampler.interior(), sampler.boundary(), sampler.initial()]

            loss, metrics = self._compute_loss(model, eqn, trainloader, model_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()

            if it % 100 == 0:
                self._print_training_info(it, iterations, loss, metrics, optimizer, model_type)

        torch.save(model.state_dict(), f'model_saved/model_{model_type}_params.pkl')

    def I_iteration(self):
        """Train radiation intensity model"""
        self._train_loop(self.model_I, equation_I.LinearTransport, "I")

    def rho_iteration(self):
        """Train material density model"""
        self._train_loop(self.model_rho, equation_rho.LinearTransport, "rho")

    def _plot_validation(self, pred, ref, ylabel: str, filename: str, Iter):
        """Visualize validation results"""
        plt.figure(figsize=(8, 5))
        plt.plot(self.ref_x.cpu().numpy(), pred, 'ro', markersize=5, label='VEF-PINNs')
        plt.plot(self.ref_x.cpu().numpy(), ref, 'k-', linewidth=1.0, label='Reference')
        plt.grid(True)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} Comparison at t=0.5")
        plt.savefig(f'./figure/{filename}_{Iter}.pdf')
        plt.close()

    def validate(self, model, ref_data, model_type: str, Iter):
        """Perform model validation"""
        with torch.no_grad():
            inputs = torch.cat([self.ref_t, self.ref_x], -1)

            if model_type == "I":
                eqn = equation_I.LinearTransport(config=self.Config, sol=model)
                pred = eqn.rho(sol=model, inputs=(self.ref_t, self.ref_x)).cpu().numpy()
            elif model_type == "rho":
                pred = model(inputs).cpu().numpy()

            error = np.sqrt(np.mean((ref_data.numpy() - pred) ** 2) / np.mean(ref_data.numpy() ** 2))

            labels = {
                "I": ("rho", "appro_rho_for_f"),
                "rho": ("rho", "appro_rho_for_rho")
            }
            self._plot_validation(pred, ref_data.numpy(), *labels[model_type], Iter)

            return error, pred

        # Drawing of the final result, modified to suit the case
    def final_plot(self, model, model_type: str):
        ref_path1 = os.path.join(os.path.abspath("../.."), "data/e2_ref_kn1e0_t=0.npz")
        ref_data1 = np.load(ref_path1)
        ref_rho1 = torch.Tensor(ref_data1["macro_frames"]).cpu().reshape(-1, 1)
        ref_path2 = os.path.join(os.path.abspath("../.."), "data/e2_ref_kn1e0_t=01.npz")
        ref_data2 = np.load(ref_path2)
        ref_rho2 = torch.Tensor(ref_data2["macro_frames"]).cpu().reshape(-1, 1)

        ref_t1 = torch.full_like(self.ref_x, 0.0)
        ref_t2 = torch.full_like(self.ref_x, 0.1)

        inputs1 = torch.cat([ref_t1, self.ref_x], -1)
        inputs2 = torch.cat([ref_t2, self.ref_x], -1)
        with torch.no_grad():
            if model_type == "I":
                eqn = equation_I.LinearTransport(config=self.Config, sol=model)
                pred1 = eqn.rho(sol=model, inputs=(ref_t1, self.ref_x)).cpu().numpy()
                pred2 = eqn.rho(sol=model, inputs=(ref_t2, self.ref_x)).cpu().numpy()
            elif model_type == "rho":
                pred1 = model(inputs1).cpu().numpy()
                pred2 = model(inputs2).cpu().numpy()

        fig = plt.figure()

        plt.plot(self.ref_x.cpu().detach().numpy(), pred1, color='b', marker='*', linestyle='--', linewidth=1.0,
                 markersize=5, markevery=50,
                 label='VEF-PINNs(t = 0)')
        plt.plot(self.ref_x.cpu().detach().numpy(), ref_rho1, color='g', linewidth=1.0, markersize=10, label='Ref(t = 0)')

        plt.plot(self.ref_x.cpu().detach().numpy(), pred2, color='r', marker='o', linestyle='--', linewidth=1.0,
                 markersize=5, markevery=50,
                 label='VEF-PINNs(t = 0.1)')
        plt.plot(self.ref_x.cpu().detach().numpy(), ref_rho2, color='k', linewidth=1.0, markersize=10, label='Ref(t = 0.1)')

        plt.legend()
        plt.xlabel(r"x")
        plt.ylabel(r"$\rho$")
        plt.title(r"$\rho,$ ref at $t = 0, 0.1$")
        plt.savefig('./figure/VEF-PINNs_{}.pdf'.format(model_type))
        plt.show()

        error1 = torch.sqrt(
            torch.mean((ref_rho1.cpu() - pred1) ** 2)
            / (torch.mean(ref_rho1.cpu() ** 2) + 1e-15)
        ).numpy()
        error2 = torch.sqrt(
            torch.mean((ref_rho2.cpu() - pred2) ** 2)
            / (torch.mean(ref_rho2.cpu() ** 2) + 1e-15)
        ).numpy()
        return  error1, error2

def main():
    """Main training workflow"""
    pinn = VEF_PINN()
    start_time = time.time()
    approx_rho_n = approx_rho_rho_n = 0

    for Iter in range(5):  # Outer alternating training loop
        print(f"\n=== Starting Alternating Training Cycle {Iter + 1}/5 ===")

        # Train I model
        pinn.I_iteration()
        error_I, approx_rho_I = pinn.validate(pinn.model_I, pinn.ref_rho, "I", Iter)

        # Check convergence
        iter_error_1 = np.max(np.abs(approx_rho_n - approx_rho_I) / (approx_rho_I + 1e-15) )
        iter_error_2 = np.max(np.abs(approx_rho_rho_n - approx_rho_I) / (approx_rho_I + 1e-15))
        iter_error = min(iter_error_1, iter_error_2)

        if iter_error < 5e-3:
            break

        # Train rho model
        pinn.rho_iteration()
        error_rho, approx_rho_rho = pinn.validate(pinn.model_rho, pinn.ref_rho, "rho", Iter)

        # Check convergence
        iter_error_1 = np.max(np.abs(approx_rho_n - approx_rho_rho) / (approx_rho_rho + 1e-15) )
        iter_error_2 = np.max(np.abs(approx_rho_rho_n - approx_rho_rho) / (approx_rho_rho + 1e-15))
        iter_error = min(iter_error_1, iter_error_2)

        if iter_error < 5e-3:
            break

        # Update references
        approx_rho_n = approx_rho_I
        approx_rho_rho_n = approx_rho_rho

        print(f"Cycle {Iter + 1} Errors: I={error_I:.2e}, rho={error_rho:.2e}")

    error_I1, error_I2 = pinn.final_plot(pinn.model_I, "I")
    error_rho1, error_rho2 = pinn.final_plot(pinn.model_rho, "rho")

    print(f"Errors of model_I for t=0 : {error_I1:.2e} and  for t=0.1 : {error_I2:.2e}")
    print(f"Errors of model_rho for t=0 : {error_rho1:.2e} and  for t=0.1 : {error_rho2:.2e}")

    print(f"\nTotal Training Time: {time.time() - start_time:.2f}s")



if __name__ == "__main__":
    main()