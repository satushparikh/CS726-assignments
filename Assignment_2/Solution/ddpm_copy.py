import torch
import torch.utils
import torch.utils.data
from tqdm.auto import tqdm
import torch.nn as nn
import torch.nn.functional as F
import argparse
import utils
import dataset
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class NoiseScheduler():
    """
    Noise scheduler for the DDPM model

    Args:
        num_timesteps: int, the number of timesteps
        type: str, the type of scheduler to use
        **kwargs: additional arguments for the scheduler

    This object sets up all the constants like alpha, beta, sigma, etc. required for the DDPM model
    
    """
    def __init__(self, num_timesteps=50, type="linear", **kwargs):

        self.num_timesteps = num_timesteps
        self.type = type

        if type == "linear":
            self.init_linear_schedule(**kwargs)
        elif type == "cosine":
            self.init_cosine_schedule(**kwargs)
        elif type == "sigmoid":
            self.init_sigmoid_schedule(**kwargs)
        else:
            raise NotImplementedError(f"{type} scheduler is not implemented") # change this if you implement additional schedulers


    def init_linear_schedule(self, beta_start, beta_end):
        """
        Initialise alpha and beta for linear noise scheduler

        Args:
            beta_start: Starting value of beta (the initial noise)
            beta_end: The final value of beta (that is addded at the last time-step)
        """

        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps, dtype=torch.float32)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alphas_cumprod)

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)

class DDPM(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.time_embedding = nn.Sequential(
            nn.Embedding(1000, 256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.model = nn.Sequential(
            nn.Linear(ndim + 256, 512),
            nn.LayerNorm(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Linear(512, ndim)
        )
    
    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x = torch.cat([x, t_emb], dim=-1)
        return self.model(x)

def train_ddpm(model, dataloader, scheduler, epochs=10, lr=1e-4, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x0, _) in enumerate(dataloader):
            x0 = x0.float().to(device)  # ensure tensor is on the correct device
            t = torch.randint(1, scheduler.num_timesteps, (x0.shape[0],)).long().to(device)
            eps = torch.randn_like(x0).to(device)

            xt = scheduler.sqrt_alpha_cumprod[t].view(-1, 1) * x0 + scheduler.sqrt_one_minus_alpha_cumprod[t].view(-1, 1) * eps

            eps_pred = model(xt, t)
            loss = mse_loss(eps, eps_pred)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Print loss for every 100th batch
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}/{len(dataloader)}], Loss: {loss.item()}")

        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss / len(dataloader)}")

def sample_ddpm(model, scheduler, num_samples=10):
    x = torch.randn((num_samples, model.model[0].in_features - 256))
    for t in reversed(range(1, scheduler.num_timesteps)):
        t_tensor = torch.tensor([t] * num_samples).long()
        eps_pred = model(x, t_tensor)
        alpha_t = scheduler.alphas[t].view(-1, 1)
        alpha_cumprod_t = scheduler.alphas_cumprod[t].view(-1, 1)
        x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * eps_pred) / torch.sqrt(alpha_t)
        if t > 1:
            z = torch.randn_like(x)
            sigma_t = torch.sqrt((1 - alpha_t) / (1 - alpha_cumprod_t))
            x += sigma_t * z
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'sample'], default='sample')
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--lbeta", type=float, default=1e-4)
    parser.add_argument("--ubeta", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_dim", type=int, default=3)
    args = parser.parse_args()

    utils.seed_everything(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_name = f'exps/ddpm_{args.n_dim}_{args.n_steps}_{args.lbeta}_{args.ubeta}_{args.dataset}'
    os.makedirs(run_name, exist_ok=True)

    model = DDPM(ndim=args.n_dim).to(device)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="linear", beta_start=args.lbeta, beta_end=args.ubeta)
    
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)

        dataset = TensorDataset(data_X, data_y)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        train_ddpm(model, dataloader, noise_scheduler, args.epochs, args.lr, args.batch_size)
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample_ddpm(model, noise_scheduler, args.n_samples)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
        print("Sampling complete, saved results.")
    else:
        raise ValueError(f"Invalid mode {args.mode}")