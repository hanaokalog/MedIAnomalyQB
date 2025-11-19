
import torch.nn
import torch.distributions

class QuasiBinarizingLayer(torch.nn.Module):
    def __init__(
        self, 
        latent_size, 
        epsilon_per_dimension = 0.2
    ):
        super().__init__()
        self.epsilon = epsilon_per_dimension
        self.latent_size = latent_size
        self.sigmoid = torch.nn.Sigmoid()
        self.laplace = torch.distributions.laplace.Laplace(0., 1. / (epsilon_per_dimension+1.0e-20))
        self.relu = torch.nn.ReLU()

    def forward(self, x):

        if 0 == self.epsilon:
            return {"x":x, 
                    "expected_firing_rate":torch.zeros_like(x).mean(dim=1), 
                    "real_firing_rate":torch.zeros_like(x).mean(dim=1)
                    }

        # asserts
        assert x.dim() == 2
        assert x.shape[1] == self.latent_size

        # clip into [0,1] by sigmoid
        x = torch.sigmoid(x)
#        x = torch.clamp(x, 0, 1)
        
        # unnoised_x
        unnoised_x = x # .clone()

        # record real neuron firing rate
        real_firing_rate = torch.where(x > 0.5, 1.0, 0.0).mean(dim=1)

        # quasi-binarize (1-bit information bottle-neck) by Laplace mechanism
        # using reparametrization trick
        if self.training:
            x = x + self.laplace.rsample(x.shape).to(x.device)

        # calculate the expected value of neuronal firing rate for each sample in minibatch
#        clampedx = torch.clamp(x, 0.0, 1.0)
#        entropy = -(clampedx * torch.log(clampedx + 1e-20) + (1.0 - clampedx) * torch.log(1.0 - clampedx + 1e-20))

        # modification for erasing bad impact of negative values
#        x = torch.relu(x)

#       expected firing rate (differentiatable!)        
        expected_firing_rate = torch.mean(x, dim=1)

        return {
            "x": x, 
            "unnoised_x": unnoised_x,
            "expected_firing_rate": expected_firing_rate,
            "real_firing_rate": real_firing_rate
        }



