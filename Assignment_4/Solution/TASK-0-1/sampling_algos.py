import os
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time

# Other settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUTPUT_PATH = "./output"  # Directory to save results (samples, plots, etc.)

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- Define two classes for Algo-1 and Algo-2 ---
##################################################
# Your code for Task-1 goes here
class Algo1_Sampler:
    def __init__(self, model, tau, burn_in, num_samples, device):
        self.model = model.to(device)
        self.tau = tau
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.device = device

    def sample(self, x0):
        samples = []
        x = x0.to(self.device)
        for t in range(self.num_samples):
            x.requires_grad = True
            # Compute gradient of E(X)
            energy = self.model(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]

            # Sample noise
            noise = torch.randn_like(x)

            # Propose new sample
            x_prime = x - (self.tau / 2) * grad + torch.sqrt(torch.tensor(self.tau)) * noise

            # Detach x_prime and set requires_grad = True
            x_prime = x_prime.detach()
            x_prime.requires_grad = True

            # Compute gradients at proposal
            energy_prime = self.model(x_prime)
            grad_prime = torch.autograd.grad(energy_prime.sum(), x_prime)[0]

            # Compute acceptance probability
            log_q_x_given_x_prime = - (1 / (4 * self.tau)) * torch.norm(x - (x_prime - (self.tau / 2) * grad_prime))**2
            log_q_x_prime_given_x = - (1 / (4 * self.tau)) * torch.norm(x_prime - (x - (self.tau / 2) * grad))**2
            alpha = torch.exp(energy - energy_prime + log_q_x_given_x_prime - log_q_x_prime_given_x).clamp(max=1)

            # Accept or reject
            if torch.rand(1).item() < alpha.item():
                x = x_prime.detach()
            samples.append(x.cpu().detach().numpy())

        # Discard burn-in samples
        return np.array(samples[self.burn_in:])

class Algo2_Sampler:
    def __init__(self, model, tau, burn_in, num_samples, device):
        self.model = model.to(device)
        self.tau = tau
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.device = device

    def sample(self, x0):
        samples = []
        x = x0.to(self.device)
        for t in range(self.num_samples):
            x = x.detach()  # Detach x from the computation graph
            x.requires_grad = True  # Enable gradient computation for x

            # Compute gradient of E(X)
            energy = self.model(x)
            grad = torch.autograd.grad(energy.sum(), x)[0]

            # Sample noise
            noise = torch.randn_like(x)

            # Update sample
            x = x - (self.tau / 2) * grad + torch.sqrt(torch.tensor(self.tau)) * noise
            samples.append(x.cpu().detach().numpy())

        # Discard burn-in samples
        return np.array(samples[self.burn_in:])

# --- Main Execution ---
if __name__ == "__main__":
    # Load pre-trained model (NN(X) = E(X))
    from get_results import EnergyRegressor, FEAT_DIM, DEVICE
    model = EnergyRegressor(FEAT_DIM)
    model.load_state_dict(torch.load("../trained_model_weights.pth", map_location=DEVICE))
    model.eval()

    # Initialize parameters
    tau = 0.1  # Step size
    burn_in = 1000  # Burn-in period
    num_samples = 5000  # Total number of samples
    x0 = torch.randn(1, FEAT_DIM)  # Initial sample

    # Run Algo-1
    algo1 = Algo1_Sampler(model, tau, burn_in, num_samples, DEVICE)
    start_time = time.time()
    samples_algo1 = algo1.sample(x0)
    algo1_time = time.time() - start_time
    print(f"Algo-1 Sampling Time: {algo1_time:.2f}s")

    # Run Algo-2
    algo2 = Algo2_Sampler(model, tau, burn_in, num_samples, DEVICE)
    start_time = time.time()
    samples_algo2 = algo2.sample(x0)
    algo2_time = time.time() - start_time
    print(f"Algo-2 Sampling Time: {algo2_time:.2f}s")

    # Save samples
    np.save(os.path.join(OUTPUT_PATH, "samples_algo1.npy"), samples_algo1)
    np.save(os.path.join(OUTPUT_PATH, "samples_algo2.npy"), samples_algo2)

    # Reshape samples to 2D
samples_algo1_reshaped = samples_algo1.reshape(samples_algo1.shape[0], -1)
samples_algo2_reshaped = samples_algo2.reshape(samples_algo2.shape[0], -1)

# Visualization using t-SNE
tsne = TSNE(n_components=2, random_state=42)
samples_2d_algo1 = tsne.fit_transform(samples_algo1_reshaped)
samples_2d_algo2 = tsne.fit_transform(samples_algo2_reshaped)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(samples_2d_algo1[:, 0], samples_2d_algo1[:, 1], s=1, alpha=0.7)
plt.title("Algo-1 Samples (t-SNE)")

plt.subplot(1, 2, 2)
plt.scatter(samples_2d_algo2[:, 0], samples_2d_algo2[:, 1], s=1, alpha=0.7)
plt.title("Algo-2 Samples (t-SNE)")

plt.savefig(os.path.join(OUTPUT_PATH, "samples_visualization.png"))
plt.show()