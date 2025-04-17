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

##################################################
# Algo 1: Metropolis-adjusted Langevin Algorithm (MALA)
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

##################################################
# Algo 2: Unadjusted Langevin Algorithm (ULA)
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
    

##################################################
# Algo 3: Hamiltonian Monte Carlo (HMC)
class Algo3_HMC_Sampler:
    def __init__(self, model, epsilon, L, burn_in, num_samples, device):
        self.model = model.to(device)
        self.epsilon = epsilon
        self.L = L
        self.burn_in = burn_in
        self.num_samples = num_samples
        self.device = device

    def sample(self, x0):
        samples = []
        x = x0.to(self.device)

        for t in range(self.num_samples):
            p = torch.randn_like(x)
            x_new = x.clone().detach().requires_grad_(True)
            p_new = p.clone()

            energy = self.model(x_new)
            grad = torch.autograd.grad(energy.sum(), x_new)[0]
            p_new = p_new - 0.5 * self.epsilon * grad

            for i in range(self.L):
                x_new = x_new + self.epsilon * p_new
                x_new = x_new.detach().requires_grad_(True)
                if i != self.L - 1:
                    energy = self.model(x_new)
                    grad = torch.autograd.grad(energy.sum(), x_new)[0]
                    p_new = p_new - self.epsilon * grad

            energy = self.model(x_new)
            grad = torch.autograd.grad(energy.sum(), x_new)[0]
            p_new = p_new - 0.5 * self.epsilon * grad
            p_new = -p_new

            current_H = self.model(x).sum() + 0.5 * torch.sum(p ** 2)
            proposed_H = energy.sum() + 0.5 * torch.sum(p_new ** 2)

            alpha = torch.exp(current_H - proposed_H).clamp(max=1)
            if torch.rand(1).item() < alpha.item():
                x = x_new.detach()
            samples.append(x.cpu().detach().numpy())

        return np.array(samples[self.burn_in:])

##################################################
# Main Execution
if __name__ == "__main__":
    from get_results import EnergyRegressor, FEAT_DIM, DEVICE
    model = EnergyRegressor(FEAT_DIM)
    model.load_state_dict(torch.load("../trained_model_weights.pth", map_location=DEVICE))
    model.eval()

    tau = 0.1
    burn_in = 1000
    num_samples = 5000
    x0 = torch.randn(1, FEAT_DIM)

    # Algo 1
    algo1 = Algo1_Sampler(model, tau, burn_in, num_samples, DEVICE)
    start_time = time.time()
    samples_algo1 = algo1.sample(x0)
    print(f"Algo-1 Time: {time.time() - start_time:.2f}s")

    # Algo 2
    algo2 = Algo2_Sampler(model, tau, burn_in, num_samples, DEVICE)
    start_time = time.time()
    samples_algo2 = algo2.sample(x0)
    print(f"Algo-2 Time: {time.time() - start_time:.2f}s")

    # Algo 3 (HMC)
    epsilon = 0.05  # Leapfrog step size
    L = 10          # Number of leapfrog steps
    algo3 = Algo3_HMC_Sampler(model, epsilon, L, burn_in, num_samples, DEVICE)
    start_time = time.time()
    samples_algo3 = algo3.sample(x0)
    print(f"Algo-3 (HMC) Time: {time.time() - start_time:.2f}s")

    # Save all samples
    np.save(os.path.join(OUTPUT_PATH, "samples_algo1.npy"), samples_algo1)
    np.save(os.path.join(OUTPUT_PATH, "samples_algo2.npy"), samples_algo2)
    np.save(os.path.join(OUTPUT_PATH, "samples_algo3.npy"), samples_algo3)

    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=SEED)
    samples_2d_algo1 = tsne.fit_transform(samples_algo1.reshape(samples_algo1.shape[0], -1))
    samples_2d_algo2 = tsne.fit_transform(samples_algo2.reshape(samples_algo2.shape[0], -1))
    samples_2d_algo3 = tsne.fit_transform(samples_algo3.reshape(samples_algo3.shape[0], -1))

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(samples_2d_algo1[:, 0], samples_2d_algo1[:, 1], s=1)
    plt.title("Algo-1 (MALA)")

    plt.subplot(1, 3, 2)
    plt.scatter(samples_2d_algo2[:, 0], samples_2d_algo2[:, 1], s=1)
    plt.title("Algo-2 (ULA)")

    plt.subplot(1, 3, 3)
    plt.scatter(samples_2d_algo3[:, 0], samples_2d_algo3[:, 1], s=1)
    plt.title("Algo-3 (HMC)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "tsne_comparison.png"))
    plt.show()