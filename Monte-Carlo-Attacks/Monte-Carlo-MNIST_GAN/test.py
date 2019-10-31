import tensorflow as tf
import numpy as np
import ais
import matplotlib.pyplot as plt
from priors import NormalPrior
from kernels import ParsenDensityEstimator
from scipy.stats import norm


class Generator(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, z):
        return z * 2 + 3


generator = Generator(1, 1)
prior = NormalPrior()
kernel = ParsenDensityEstimator()
model = ais.Model(generator, prior, kernel, 1000, 10000)

#p = norm()
#x = np.linspace(norm.ppf(0.01, loc=3, scale=2), norm.ppf(0.99, loc=3, scale=2), 100)
#p1 = norm.pdf(x, loc=3, scale=2)
#xx = np.reshape(x, [100, 1])

num_samples = 1
xx = np.reshape([2], [num_samples, 1])
schedule = ais.get_schedule(num_samples, rad=4)

p2 = model.ais(xx, schedule)

print('results')
print(p2)
print(np.exp(p2))
