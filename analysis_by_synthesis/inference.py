import numpy as np
import torch
from torch import nn, optim

from .loss_functions import samplewise_loss_function
from .custom_kernel import samplewise_loss_function as pairwise_loss_function
from .utils import auto_batch


class RobustInference(nn.Module):
    """Takes a trained ABS model and replaces its variational inference
    with robust inference."""

    def __init__(self, abs_model, device, n_samples, n_iterations, *, fraction_to_dismiss, lr, radius, custom_kernel):
        super().__init__()

        self.abs = abs_model
        self.vaes = abs_model.vaes
        self.lr = lr
        self.beta = abs_model.beta
        self.radius = radius
        self.custom_kernel = custom_kernel
        self.name = f'{n_samples}_{n_iterations}'

        # create a set of random latents that we will reuse
        n_latents = self.vaes[0].n_latents
        self.z = self.draw_random_latents(n_samples, n_latents, fraction_to_dismiss).to(device)

        # assuming that z's were sampled from a normal distribution with mean = z, var = 1
        # note that we haven't acutally sampled z; instead z is simply mu
        self.mu = self.z
        self.logvar = torch.tensor(0.).to(device)

        self.cached_reconstructions = {}

        assert n_iterations >= 0, 'n_iterations must be non-negative'
        self.gradient_descent_iterations = n_iterations

    @staticmethod
    def draw_random_latents(n_samples, n_latents, fraction_to_dismiss):
        assert 0 <= fraction_to_dismiss < 1

        z = torch.randn(int(n_samples / (1 - fraction_to_dismiss)), n_latents, 1, 1)

        if z.size()[0] > n_samples:
            # ignore the least likely samples
            d = torch.sum(z ** 2, dim=(1, 2, 3))
            _, best = torch.sort(d)
            best = best[:n_samples]
            z = z[best]

        return z

    @staticmethod
    def clip_to_sphere_(z, radius):
        """Clips latents to a sphere. Operates in-place!

        This function assumes that the shape of z is
        (n_classes, batch_size, *latents_shape)"""

        if radius == np.inf:
            return

        # flatten the latent dimensions because torch.norm only works on one
        zr = z.reshape(z.size()[:2] + (-1,))

        length = torch.norm(zr, p=2, dim=2)

        # determine latents that are larger than desired
        mask = length > radius

        # add missing singleton dimensions to the end
        length = length.view(length.size() + (1,) * (z.dim() - length.dim()))

        z[mask] = z[mask] / length[mask] * radius

    def invalidate_cache(self):
        self.cached_reconstructions = {}

    def forward(self, x):
        """This performs robust inference by finding the optimal latents for
        each VAE using optimization rather than the encoder network."""

        if self.custom_kernel:
            samplewise_loss_function_b = pairwise_loss_function
        else:
            # calculating the loss for all pairs of inputs and reconstructions
            # can be extremely memory consuming; batch_size * n_samples * input_size
            # -> we therefore wrap our loss function with auto_batch

            # determine a good batch size of the inputs given the number of
            # reconstructions (e.g. 80 or 8000)
            def get_batch_size():
                GiB = 2**30
                memory = 4 * GiB
                n_recs = len(self.z)
                input_size = int(np.prod(x.shape[-3:]))
                bytes_per_float = 4
                batch_size = int(memory / bytes_per_float / input_size / n_recs)
                return batch_size

            samplewise_loss_function_b = auto_batch(samplewise_loss_function, get_batch_size(), n_args=1)

        with torch.no_grad():
            losses = []
            recs = []
            mus = []
            for vae in self.vaes:
                # pass the random latents through the VAEs
                if vae not in self.cached_reconstructions:
                    self.cached_reconstructions[vae] = vae.decoder(self.z)
                rec = self.cached_reconstructions[vae]

                # determine the best latents for each sample in x given this VAE
                # -> add a second batch dimension to x that will be broadcasted to the number of reconstructions
                # -> add a second batch dimension to rec that will be broadcasted to the number of inputs in x
                loss = samplewise_loss_function_b(x.unsqueeze(1), rec.unsqueeze(0), self.mu, self.logvar, self.beta)
                assert loss.dim() == 2
                # take min over samples in z
                loss, indices = loss.min(dim=1)

                losses.append(loss)
                recs.append(rec[indices])
                mus.append(self.mu[indices])

            mus = torch.stack(mus)

            if self.gradient_descent_iterations > 0:
                # for each sample and VAE, try to improve the best latents
                # further using gradient descent
                mus = self.gradient_descent(x, mus)

                # update losses and recs
                recs = [vae.decoder(mu) for vae, mu in zip(self.vaes, mus)]
                losses = [samplewise_loss_function(x, rec, mu, self.logvar, self.beta)
                          for rec, mu in zip(recs, mus)]

            recs = torch.stack(recs)
            losses = torch.stack(losses)

            logits = -losses.transpose(0, 1)
            logvars = torch.zeros_like(mus)
            return logits, recs, mus, logvars

    def gradient_descent(self, x, z):
        with torch.enable_grad():
            # create a completely independent copy of z
            z = torch.tensor(z, requires_grad=True).to(z.device)
            optimizer = optim.Adam([z], lr=self.lr)

            for j in range(self.gradient_descent_iterations):
                optimizer.zero_grad()

                loss = 0
                for vae, zi in zip(self.vaes, z):
                    rec = vae.decoder(zi)
                    loss += samplewise_loss_function(x, rec, zi, self.logvar, self.beta).sum()

                loss.backward()
                optimizer.step()

                # must operate on .data because PyTorch doesn't allow
                # in-place modifications of a leaf Variable itself
                self.clip_to_sphere_(z.data, radius=self.radius)
        return z
