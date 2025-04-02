import torch
from torch.optim import AdamW
import numpy as np

class AdamCPR_WS(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, mu=0.1, kappa_type="w", s_step=500):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.mu = mu
        self.kappa_type = kappa_type.lower()
        self.s_step = s_step
        self.t = 0
        
        # Initialize λ and κ as lists of floats
        self.lambdas = [0.0] * len(self.param_groups)
        self.kappas = [None] * len(self.param_groups)
        print(f"Initialized {len(self.param_groups)} parameter groups.")
        
        # Tracking R(θ) history
        self.R_history = [[] for _ in self.param_groups]

    def R(self, p):
        """Compute regularization term for a parameter tensor"""
        with torch.no_grad():
            if self.kappa_type == "i":
                return p.norm().item()
            return (p * torch.exp(-p.abs())).norm().item()

    def step(self, closure=None):
        loss = super().step(closure)
        self.t += 1
        
        for j, group in enumerate(self.param_groups):
            # Compute average R(θ) for the group
            R_vals = []
            for p in group['params']:
                if p.grad is not None:
                    R_vals.append(self.R(p))
            
            if not R_vals:
                continue
                
            avg_R = np.mean(R_vals)
            self.R_history[j].append(avg_R)

            print("t : {self.t}, step : {self.s_step}, group : {j}, avg_R : {avg_R:.4f}")
            
            # --- Kappa Update (Warm Start) ---
            if self.t >= self.s_step and self.kappas[j] is None:
                self.kappas[j] = 2 * avg_R  # 2× as per paper
                print(f"Group {j}: κ set to {self.kappas[j]:.4f} at step {self.t}")
            
            # --- Lagrange (λ) Update ---
            if self.kappas[j] is not None:
                constraint_violation = avg_R - self.kappas[j]
                self.lambdas[j] = max(0.0, self.lambdas[j] + self.mu * constraint_violation)
                
                # Apply penalty to gradients
                for p in group['params']:
                    if p.grad is not None:
                        if self.kappa_type == "i":
                            p.grad += self.lambdas[j] * p.data
                        else:
                            kappa_w_grad = (1 - p.data.abs()) * torch.exp(-p.data.abs()) * p.data
                            p.grad += self.lambdas[j] * kappa_w_grad
        
        return loss