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
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.sqrt_alpha_cumprod)

    def init_cosine_schedule(self):
        """Initializes a cosine noise schedule as per [1]."""
        # Reference: https://www.zainnasir.com/blog/cosine-beta-schedule-for-denoising-diffusion-models/
        epsilon = 0.008
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
        f_t = torch.cos(( (steps / self.num_timesteps) + epsilon ) / ( 1 + epsilon ) * ( np.pi / 2 )) ** 2
        alphas_cumprod = f_t / f_t[0]
        self.alphas_cumprod = alphas_cumprod[:-1]
        self.betas = 1 - (self.alphas_cumprod[1:] / self.alphas_cumprod[:-1])
        self.betas = torch.clamp(self.betas, min=1e-5, max=0.999)

    def init_sigmoid_schedule(self, beta_start, beta_end):
        """Initializes a sigmoid noise schedule."""
        x = torch.linspace(-6, 6, self.num_timesteps)
        betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
        self.betas = betas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def __len__(self):
        return self.num_timesteps
    

# def linear_noise_schedule_variations():
#     lbeta_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
#     ubeta_values = [0.02, 0.03, 0.05, 0.07, 0.1]

#     results = []
#     for lbeta, ubeta in zip(lbeta_values, ubeta_values):
#         scheduler = NoiseScheduler(num_timesteps=50, type="linear", beta_start=lbeta, beta_end=ubeta)
#         results.append((lbeta, ubeta, scheduler.betas.numpy()))

#     return results

#########################################
# Helper modules for the complex U-Net  #
#########################################

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    Adapted from Fairseq.
    """
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(timesteps.shape[0], 1)], dim=1)
    return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()
    
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        # Project and add time embedding
        t_emb_proj = self.time_emb_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb_proj
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        h = self.dropout(h)
        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Using strided conv for downsampling
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Upsample via nearest-neighbor then conv
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q = self.q(x_norm).reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        k = self.k(x_norm).reshape(B, C, H * W)                     # (B, C, HW)
        v = self.v(x_norm).reshape(B, C, H * W).permute(0, 2, 1)      # (B, HW, C)
        attn = torch.bmm(q, k) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)                                    # (B, HW, C)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out

#########################################
# U-Net architecture for DDPM           #
#########################################

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, channel_mults=(1,2,4,8), num_res_blocks=2, time_emb_dim=256):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(256, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # Input convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        self.downs = []
        channels = base_channels
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResidualBlock(channels, out_ch, time_emb_dim))
                channels = out_ch
            self.down_blocks.append(Downsample(channels))
            self.downs.append(channels)
        
        # Middle blocks
        self.mid_block1 = ResidualBlock(channels, channels, time_emb_dim)
        self.mid_attn = SelfAttention(channels)
        self.mid_block2 = ResidualBlock(channels, channels, time_emb_dim)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_blocks.append(Upsample(channels))
            for _ in range(num_res_blocks + 1):  # one extra block to merge skip connection
                self.up_blocks.append(ResidualBlock(channels + out_ch, out_ch, time_emb_dim))
                channels = out_ch
        
        # Output layers
        self.norm_out = nn.GroupNorm(8, channels)
        self.act_out = nn.ReLU()
        self.conv_out = nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        """
        x: (B, in_channels, H, W)
        t: (B,) containing timesteps
        """
        # Create time embeddings
        t_emb = get_timestep_embedding(t, 256).to(x.device)
        t_emb = self.time_mlp(t_emb)
        
        hs = []
        h = self.conv_in(x)
        hs.append(h)
        # Downsampling path
        for module in self.down_blocks:
            if isinstance(module, ResidualBlock):
                h = module(h, t_emb)
                hs.append(h)
            else:  # Downsample module
                h = module(h)
                hs.append(h)
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        # Upsampling path
        for module in self.up_blocks:
            if isinstance(module, ResidualBlock):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, t_emb)
            else:  # Upsample module
                h = module(h)
        h = self.norm_out(h)
        h = self.act_out(h)
        return self.conv_out(h)

#########################################
# DDPM and related classes              #
#########################################

class DDPM(nn.Module):
    def __init__(self, in_channels=3, n_steps=200):
        super().__init__()
        self.n_dim = in_channels  # For images, this is the number of channels (e.g., 3 for RGB)
        self.n_steps = n_steps
        # Replace the simple model with the complex U-Net
        self.model = UNet(in_channels, in_channels, base_channels=64, channel_mults=(1,2,4,8), num_res_blocks=2, time_emb_dim=256)
    
    def forward(self, x, t):
        return self.model(x, t)

class ConditionalDDPM(DDPM):
    def __init__(self, in_channels=3, n_steps=200, num_classes=10):
        super().__init__(in_channels, n_steps)
        # Add a class embedding to condition the model
        self.class_embed = nn.Embedding(num_classes, 256)
    
    def forward(self, x, t, labels):
        class_emb = self.class_embed(labels)
        # Combine class embedding with time embedding by simple addition
        # (Alternatively, you can use a more sophisticated fusion method)
        t_emb = get_timestep_embedding(t, 256).to(x.device)
        t_emb = self.model.time_mlp(t_emb) + class_emb
        # Now forward pass through UNet with the combined embedding
        return self.model(x, t)

class ClassifierDDPM():
    def __init__(self, model: ConditionalDDPM, noise_scheduler):
        self.model = model
        self.noise_scheduler = noise_scheduler

    def predict(self, x):
        # Dummy implementation: adjust as needed
        logits = self.model(x, torch.zeros(x.shape[0], dtype=torch.long).to(x.device),
                            torch.zeros(x.shape[0], dtype=torch.long).to(x.device))
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x):
        logits = self.model(x, torch.zeros(x.shape[0], dtype=torch.long).to(x.device),
                            torch.zeros(x.shape[0], dtype=torch.long).to(x.device))
        return F.softmax(logits, dim=1)

#########################################
# Training and Sampling Functions       #
#########################################

def train(model, noise_scheduler, dataloader, optimizer, epochs, run_name):
    model.train()
    for epoch in range(epochs):
        for x, _ in tqdm(dataloader):
            optimizer.zero_grad()
            # Sample random timesteps for each batch element
            t = torch.randint(0, noise_scheduler.num_timesteps, (x.shape[0],)).to(x.device)
            noise = torch.randn_like(x)
            # Add noise according to the forward process
            alpha_bar = noise_scheduler.alpha_bars[t].view(-1, 1, 1, 1).to(x.device)
            x_noisy = x * alpha_bar + noise * (1 - alpha_bar).sqrt()
            predicted_noise = model(x_noisy, t)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

@torch.no_grad()
def sample(model, n_samples, noise_scheduler, return_intermediate=False): 
    model.eval()
    # Start from pure noise
    x = torch.randn(n_samples, model.n_dim, 32, 32)  # Assuming images of size 32x32; adjust as needed.
    intermediates = []
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long).to(x.device)
        predicted_noise = model(x, t_tensor)
        alpha_bar = noise_scheduler.alpha_bars[t].to(x.device)
        # Simple update rule (for illustration; in practice, use the proper reverse process equations):\n        x = (x - (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()\n        if return_intermediate:\n            intermediates.append(x.clone())\n    return x if not return_intermediate else intermediates

def sampleCFG(model, n_samples, noise_scheduler, guidance_scale, class_label):
    model.eval()
    x = torch.randn(n_samples, model.n_dim, 32, 32).to(next(model.parameters()).device)
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long).to(x.device)
        # Conditional prediction using the provided class label
        predicted_noise = model(x, t_tensor, torch.full((n_samples,), class_label, dtype=torch.long).to(x.device))
        alpha_bar = noise_scheduler.alpha_bars[t].to(x.device)
        x = (x - guidance_scale * (1 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
    return x

def sampleSVDD(model, n_samples, noise_scheduler, reward_scale, reward_fn):
    # Implement SVDD-PM sampling procedure that uses a reward function to guide sampling
    # This is left as an exercise; below is a placeholder.
    model.eval()
    x = torch.randn(n_samples, model.n_dim, 32, 32).to(next(model.parameters()).device)
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.full((n_samples,), t, dtype=torch.long).to(x.device)
        predicted_noise = model(x, t_tensor)
        # Incorporate reward function into the update step (placeholder logic)
        reward = reward_fn(x)
        alpha_bar = noise_scheduler.alpha_bars[t].to(x.device)
        x = (x - (1 - alpha_bar).sqrt() * predicted_noise * (1 + reward_scale * reward)) / alpha_bar.sqrt()
    return x

#########################################
# Main Execution                        #
#########################################

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

    # Use the complex UNet-based DDPM
    model = DDPM(in_channels=args.n_dim, n_steps=args.n_steps).to(device)
    noise_scheduler = NoiseScheduler(num_timesteps=args.n_steps, type="linear", beta_start=args.lbeta, beta_end=args.ubeta)
    
    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_X, data_y = dataset.load_dataset(args.dataset)
        data_X = data_X.to(device)
        data_y = data_y.to(device)
        dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(data_X, data_y),
                                                   batch_size=args.batch_size, shuffle=True)
        train(model, noise_scheduler, dataloader, optimizer, args.epochs, run_name)
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(f'{run_name}/model.pth'))
        samples = sample(model, args.n_samples, noise_scheduler)
        torch.save(samples, f'{run_name}/samples_{args.seed}_{args.n_samples}.pth')
    else:
        raise ValueError(f"Invalid mode {args.mode}")
