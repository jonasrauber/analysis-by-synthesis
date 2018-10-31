import torch
from torchvision.utils import make_grid


def sample(model, device, step, writer):
    if writer is None:
        return
    model.eval()
    n_latents = model.vaes[0].n_latents
    with torch.no_grad():
        zs = torch.randn(12, n_latents, 1, 1).to(device)
        samples = torch.cat([vae.decoder(zs).cpu() for vae in model.vaes])
        grid = make_grid(samples, nrow=12)
        writer.add_image(f'samples', grid, step)
