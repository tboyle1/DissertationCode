import math
import numpy as np
import decimal
import numpy.random as nrand
import matplotlib.pyplot as plt
import pandas as pd

class ModelParameters:
    """
    This provides all of the parameters for the model
    """

    def __init__(self,
                 price_start, time, time_rate, vol,
                 drift, jumps_lamda=0.0, jumps_sigma=0.0, jumps_mu=0.0,
                 cir_a=0.0, cir_mu=0.0, all_r0=0.0, cir_rho=0.0,
                 ou_a=0.0, ou_mu=0.0,
                 heston_a=0.0, heston_mu=0.0, heston_vol0=0.0):

        #starting asset value--->
        self.price_start = price_start
        self.time = time
        self.time_rate = time_rate
        self.vol = vol
        self.drift = drift
        self.jumps_lamda = jumps_lamda
        self.jumps_sigma = jumps_sigma
        self.jumps_mu = jumps_mu
        self.cir_a = cir_a
        self.cir_mu = cir_mu
        self.all_r0 = all_r0
        self.cir_rho = cir_rho
        self.ou_a = ou_a
        self.ou_mu = ou_mu
        self.heston_a = heston_a
        self.heston_mu = heston_mu
        self.heston_vol0 = heston_vol0


def convert_to_returns(log_returns):
    """
    Exponentiates log returns
    :param log_returns: the log returns to be exponentiated
    :return: the exponential returns
    """
    return np.exp(log_returns)

def convert_to_prices(param, log_returns):
    """
    Takes log returns and coverts them into normal returns and then computes a price sequence from the starting price ''price_start''
    :param param: model parameters
    :param log_returns: the returns to be made into an exponential
    :return:
    """
    returns = convert_to_returns(log_returns)
    #sequence of prices with price of ''price_start''
    price_sequence = [param.price_start]
    for i in range(1, len(returns)):
        #add the at t-1 * return at t
        price_sequence.append(price_sequence[i-1]*returns[i-1])
    return np.array(price_sequence)

def brownian_motion_log_returns(param):
    """
    Returns a brownian motion process
    :param param: model parameters
    :return: returns a price which follows the brownian motion process
    """
    sqrt_delta_sigma = math.sqrt(param.time_rate) * param.vol
    return nrand.normal(loc=0, scale=sqrt_delta_sigma, size=param.time)

def brownian_motion_levels(param):
    """
    Returns a price sequence whose returns follow a brownian motion process
    :param param: model parameters
    :return: returns a brownian motion process
    """
    return convert_to_prices(param, brownian_motion_log_returns(param))

def geometric_brownian_motion_log_returns(param):
    """
    Produces a sequence of log returns through a stochastic process which fits within a random geometric brownian motion distribution
    :param param: model parameters
    :return: returns log returns of a stochastic process
    """
    assert isinstance(param, ModelParameters)
    gbmprocess = np.array(brownian_motion_log_returns(param))
    sigma_pow_mu_delta = (param.drift - 0.5 * math.pow(param.vol, 2.0)) * param.time_rate
    return gbmprocess + sigma_pow_mu_delta

def geometric_brownian_motion_levels(param):
    """
    Returns price levels which fit within the geometric brownian motion process
    :param param: model parameters
    :return: price levels for the asset
    """
    return convert_to_prices(param, geometric_brownian_motion_log_returns(param))

def plot_stochastic_process(processes, title):
    """
    Plots the process with a specific title
    :param processes: GBM or BM
    :param title: User input
    :return: plots a graph
    """
    plt.style.use(['bmh'])
    fig, ax = plt.subplots(1)
    fig.suptitle(title, fontsize=16)
    ax.set_xlabel('Time, t')
    ax.set_ylabel('Simulated Asset Price')
    x_axis = np.arange(0, len(processes[0]),1)
    for i in range(len(processes)):
        plt.plot(x_axis, processes[i])
    plt.show()

Drift = 0.05
Vol = 0.2
paths = 1000

mp = ModelParameters(price_start=1,
                     time=1512,
                     time_rate=0.003968254,
                     drift=Drift,
                     vol=Vol,
                     jumps_lamda=0.00125,
                     jumps_sigma=0.001,
                     jumps_mu=-0.2,
                     cir_a=3.0,
                     cir_mu=0.5,
                     cir_rho=0.5,
                     ou_a=3.0,
                     heston_a=0.25,
                     heston_mu=0.35,
                     heston_vol0=0.06125)


def RathbonesExampleBrownianMotion(paths=paths):
    brownian_motion_example = []
    for i in range(paths):
        brownian_motion_example.append(brownian_motion_levels(mp))
    plot_stochastic_process(brownian_motion_example,'Brownian Motion Example')

def RathbonesExampleGeometricBrownianMotion(paths=paths):
    geometric_brownian_motion_example = []
    for i in range(paths):
        geometric_brownian_motion_example.append(geometric_brownian_motion_levels(mp))
    plot_stochastic_process(geometric_brownian_motion_example, 'Geometric Brownian Motion')

output = []
for i in range(paths):
    output.append(geometric_brownian_motion_levels(mp))
plot_stochastic_process(output, 'GBM Returns')






