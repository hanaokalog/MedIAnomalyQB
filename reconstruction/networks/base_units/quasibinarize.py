
import torch.nn
import torch.distributions

class QuasiBinarizingLayer(torch.nn.Module):
    def __init__(
        self, 
        latent_size, 
        epsilon_per_dimension = 0.2,
        using_heaviside = False,
        adding_noise_in_test = False
    ):
        super().__init__()
        self.epsilon = epsilon_per_dimension
        self.latent_size = latent_size
        self._using_heaviside = using_heaviside
        self._adding_noise_in_test = adding_noise_in_test
        self.sigmoid = torch.nn.Sigmoid()
        self.laplace = torch.distributions.laplace.Laplace(0., 1. / (epsilon_per_dimension+1.0e-20))
        self.relu = torch.nn.ReLU()

    @property
    def using_heaviside(self):
        return self._using_heaviside
    
    @using_heaviside.setter
    def using_heaviside(self, value):
        assert isinstance(value, bool)
        self._using_heaviside = value

    @property
    def adding_noise_in_test(self):
        return self._adding_noise_in_test
    
    @adding_noise_in_test.setter
    def adding_noise_in_test(self, value):
        assert isinstance(value, bool)
        self._adding_noise_in_test = value

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
#        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        
        # unnoised_x
        unnoised_x = x # .clone()

        # record real neuron firing rate
        real_firing_rate = torch.where(x > 0.5, 1.0, 0.0).mean(dim=1)
        
        # quasi-binarize (1-bit information bottle-neck) by Laplace mechanism
        # using reparametrization trick?? <- no meaning because the probability distribution to be sampled is not dependent upon any network parameter
        if self.training or self.adding_noise_in_test:
            x = x + self.laplace.sample(x.shape).to(x.device)

        # expected firing rate (differentiatable!)        
        expected_firing_rate = torch.mean(x, dim=1)

        # heavisided (binarized) result (undifferntiatable)
        if(self.using_heaviside):
            x = torch.where(x > 0.5, 1.0, 0.0)

        return {
            "x": x, 
            "unnoised_x": unnoised_x,
            "expected_firing_rate": expected_firing_rate,
            "real_firing_rate": real_firing_rate
        }



