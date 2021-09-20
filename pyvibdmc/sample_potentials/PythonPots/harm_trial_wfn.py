import pyvibdmc as pv
import numpy as np
import matplotlib.pyplot as plt


def harmonic_oscillator(x, mass, omega):
    alpha = mass * omega
    psi = (alpha / np.pi) ** 0.25 * np.exp(-alpha * x ** 2 / 2)
    return psi


def trial_harm(x):
    """Must return num_walkers array"""
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    return harmonic_oscillator(x, mass, omega).squeeze()


def first_derivative(x):
    """Must return num_walkers x num_atoms x 3 array, or in this case num_walkers x 1 x 1"""
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    alpha = mass * omega
    derv = (alpha / np.pi) ** 0.25 * (-alpha * x) * np.exp(-alpha * x ** 2 / 2)
    return derv


def second_derivative(x):
    """Must return num_walkers x num_atoms x 3 array, or in this case num_walkers x 1 x 1"""
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    alpha = mass * omega
    sderv = (alpha / np.pi) ** 0.25 * (alpha ** 2 * x ** 2 - alpha) * np.exp(-alpha * x ** 2 / 2)
    return sderv

def derivative(x):
    trl = trial_harm(x)
    derv = first_derivative(x)
    sderv = second_derivative(x)
    return derv / trl[:,np.newaxis,np.newaxis], sderv / trl[:,np.newaxis,np.newaxis]

if __name__ == '__main__':
    x = np.linspace(-1, 1, 100)
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    y = trial_harm(x)
    y1 = first_derivative(x)
    y2 = second_derivative(x)
    plt.plot(x, y)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
