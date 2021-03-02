import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from botorch.acquisition import acquisition 

class ProximalAcqusitionFunction(acquisition.AcquisitionFunction):
    def __init__(self, model, sigma_matrix):
        '''
        Acquisition function that biases other acquistion functions towards a
        nearby region in input space

        Arguments
        ---------
        model : Model
            A fitted model
        
        precision_matrix : torch.tensor, shape (D x D)
            Precision matrix used in the biasing multivariate distribution, D is 
            the dimensionality of input space

        '''
        
        super().__init__(model)

        self.register_buffer('sigma_matrix', sigma_matrix)

    def forward(self, X):
        #get the last point in the training set (assumed to be the most
        #recently evaluated point)
        last_pt = self.model.train_inputs[0][-1].double()

        
        #define multivariate normal
        d = MultivariateNormal(last_pt, self.sigma_matrix.double())

        #use pdf to calculate the weighting - normalized to 1 at the last point
        norm = torch.exp(d.log_prob(last_pt).flatten())
        weight = torch.exp(d.log_prob(X).flatten()) / norm
        
        return weight
