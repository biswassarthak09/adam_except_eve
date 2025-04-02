import torch
from torch.optim import AdamW

class AdamCPR_IP(AdamW):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, mu=0.05, kappa_type="w"):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.mu = mu
        self.kappa_type = kappa_type.lower()
        self.t = 0
        
        # Initialize λ and κ
        self.lambdas = [0.0 for _ in self.param_groups]
        self.kappas = [None for _ in self.param_groups]
        
        # For inflection point detection
        self.prev_R = [None for _ in self.param_groups]      # R(θ) at t-1
        self.prev_dR = [None for _ in self.param_groups]     # First derivative (dR/dt)
        self.prev_d2R = [None for _ in self.param_groups]    # Second derivative (d²R/dt²)

    def R(self, p):
        if self.kappa_type == "i":
            return p.data.norm()
        return (p.data * torch.exp(-p.data.abs())).norm()

    def step(self, closure=None):
        loss = super().step(closure)
        self.t += 1
        
        for j, group in enumerate(self.param_groups):
            if not group['params']:
                continue
                
            # Compute average R(θ) for the group
            total_R = 0.0
            param_count = 0
            for p in group['params']:
                if p.grad is not None:
                    total_R += self.R(p)
                    param_count += 1
            
            if param_count == 0:
                continue
                
            avg_R = total_R / param_count
            
            # --- Inflection Point Detection ---
            # First derivative (slope)
            current_dR = avg_R - self.prev_R[j] if self.prev_R[j] is not None else 0
            
            # Second derivative (change of slope)
            if self.prev_dR[j] is not None:
                current_d2R = current_dR - self.prev_dR[j]
            else:
                current_d2R = 0
            
            # Check for inflection point (second derivative turns negative)
            if self.prev_d2R[j] is not None and current_d2R < 0 and self.kappas[j] is None:
                self.kappas[j] = avg_R  # κ = R(θ) at inflection point
                print(f"Group {j}: κ set to {self.kappas[j]:.4f} at step {self.t}")
            
            # --- Lagrange (λ) Update ---
            if self.kappas[j] is not None:
                constraint_violation = avg_R - self.kappas[j]
                self.lambdas[j] = max(0, self.lambdas[j] + self.mu * constraint_violation)
                
                # Apply penalty to gradients
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if self.kappa_type == "i":
                        p.grad += self.lambdas[j] * p.data
                    else:
                        grad = (1 - p.data.abs()) * torch.exp(-p.data.abs()) * p.data
                        p.grad += self.lambdas[j] * grad
            
            # Update history
            self.prev_d2R[j] = current_d2R
            self.prev_dR[j] = current_dR
            self.prev_R[j] = avg_R
        
        return loss