import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_gradient_dynamics(mu_val=5000.0):
    """
    Generates a scientific plot comparing gradient flow for Linear vs. Mu-Law
    in the critical 'Dark Regime' (normalized inputs 0.0 to 0.1).
    """
    print(f"Generating Gradient Analysis for mu={mu_val}...")

    # 1. Simulate Input Signal (Dark to Mid-tone)
    # Range [0, 0.1] covers the bottom 10% of dynamic range where Deep Learning usually fails
    # We use requires_grad=True to track the backward pass
    x = torch.linspace(0.0001, 0.1, 1000, requires_grad=True)

    # --- Model A: Linear Scaling (Baseline / Standard CNN) ---
    # y = x
    y_lin = x
    # Simulate Loss: We want to maximize y (dummy objective to get gradient)
    y_lin.sum().backward()
    grad_lin = x.grad.clone()
    x.grad.zero_()  # Reset gradients

    # --- Model B: Hyper-ISP (Ours) ---
    # y = log(1 + mu*x) / log(1 + mu)
    # This is the forward pass of your DynamicMuLaw layer
    mu_tensor = torch.tensor(mu_val)
    y_mu = torch.log1p(mu_tensor * x) / torch.log1p(mu_tensor)
    y_mu.sum().backward()
    grad_mu = x.grad.clone()

    # --- Analysis ---
    x_np = x.detach().numpy()
    grad_lin_np = grad_lin.numpy()
    grad_mu_np = grad_mu.numpy()

    # Calculate the boost factor at the darkest pixel
    boost_factor = grad_mu_np[0] / grad_lin_np[0]

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # Plot Gradients
    plt.plot(x_np, grad_mu_np, label=f'Hyper-ISP (Ours, $\mu={int(mu_val)}$)', color='#d62728', linewidth=2.5)
    plt.plot(x_np, grad_lin_np, label='Linear Baseline (Standard CNN)', color='#1f77b4', linestyle='--', linewidth=2.5)

    # Styling
    plt.title('Theoretical Gradient Magnitude in Low-Light Regime', fontsize=14)
    plt.xlabel('Input Pixel Intensity (Normalized [0.0 - 0.1])', fontsize=12)
    plt.ylabel('Gradient Magnitude ($\partial y / \partial x$)', fontsize=12)

    # Log scale is crucial to show the orders of magnitude difference
    plt.yscale('log')

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)

    # Annotation
    plt.annotate(f'~{int(boost_factor)}x Gradient Boost\n(Solves Vanishing Gradient)',
                 xy=(x_np[0], grad_mu_np[0]),
                 xytext=(0.02, grad_mu_np[0] / 10),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    # Save
    output_filename = 'gradient_boost_proof.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {output_filename}")
    print(f"   Darkest Pixel Gradient (Linear): {grad_lin_np[0]:.4f}")
    print(f"   Darkest Pixel Gradient (Ours):   {grad_mu_np[0]:.4f}")
    print(f"   Boost Factor: {boost_factor:.2f}x")


if __name__ == "__main__":
    plot_gradient_dynamics()