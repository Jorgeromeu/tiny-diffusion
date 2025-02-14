from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_diffusion_model(model, dataset, noise_scheduler, config, device="cpu"):
    train_dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    pbar = tqdm(range(config.num_epochs))

    losses = []

    for epoch in pbar:
        model.train()

        epoch_loss = 0.0

        for batch in train_dataloader:
            sample = batch[0].to(device)

            # sample timesteps
            max_T = len(noise_scheduler.betas)
            timesteps = torch.randint(0, max_T, (sample.shape[0],)).long().to(device)

            # add noise to sample
            epsilon = torch.randn(sample.shape).to(device)
            noisy = noise_scheduler.add_noise(sample, epsilon, timesteps)

            # compute loss
            noise_pred = model(noisy, timesteps)

            loss = F.mse_loss(noise_pred, epsilon)

            # backprop
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            # logging
            losses.append(loss.item())
            epoch_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch} | Loss: {epoch_loss / len(train_dataloader):.4f}"
        )

    return losses


@dataclass
class DiffusionTrajectory:
    noise_levels: Tensor
    samples: Tensor


def sample_diffusion_model(
    model,
    scheduler,
    sample_shape,
    n_samples=1,
    n_inference_steps=10,
    device="cpu",
    output_trajectory=False,
    generator=None,
):
    samples = torch.randn(n_samples, *sample_shape, generator=generator)
    scheduler.set_timesteps(n_inference_steps)

    if output_trajectory:
        denoising_levels = scheduler.timesteps
        max_level = Tensor([scheduler.num_train_timesteps]).to(denoising_levels)
        all_levels = torch.cat((max_level, denoising_levels), dim=0)
        trajectory = DiffusionTrajectory(
            noise_levels=all_levels,
            samples=[samples.cpu().clone()],
        )

    for t in tqdm(scheduler.timesteps, desc="sampling"):
        timesteps = t.expand(n_samples)

        with torch.no_grad():
            # model_input = scheduler.scale_model_input(samples, t)
            model_input = samples
            noise_pred = model(model_input, timesteps)

        samples = scheduler.step(
            noise_pred, t, samples, generator=generator
        ).prev_sample

        if output_trajectory:
            trajectory.samples.append(samples.cpu().clone())

    if output_trajectory:
        trajectory.samples = torch.stack(trajectory.samples, dim=0)
        return trajectory

    return samples
