import torch
import torch.nn as nn
import torch.nn.functional as F
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

            epoch_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch} | Loss: {epoch_loss / len(train_dataloader):.4f}"
        )

    return model


def sample_diffusion_model(
    model, scheduler, sample_shape, n_samples=1, n_inference_steps=10, device="cpu"
):
    samples = torch.randn(n_samples, *sample_shape)

    scheduler.set_timesteps(n_inference_steps)

    print(scheduler.timesteps)

    for t in scheduler.timesteps:
        timesteps = t.expand(n_samples)

        with torch.no_grad():
            noise_pred = model(samples, timesteps)

        samples = scheduler.step(noise_pred, t, samples).prev_sample

    return samples
