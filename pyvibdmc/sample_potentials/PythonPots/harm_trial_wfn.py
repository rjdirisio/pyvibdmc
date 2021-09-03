import pyvibdmc as pv
import numpy as np
import matplotlib.pyplot as plt


def harmonic_oscillator(x, mass, omega):
    alpha = mass * omega
    psi = (alpha / np.pi) ** 0.25 * np.exp(-alpha * x ** 2 / 2)
    return psi


def trial_harm(x):
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    return harmonic_oscillator(x, mass, omega)


def first_derivative(x):
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    alpha = mass * omega
    derv = (alpha / np.pi) ** 0.25 * (-alpha * x) * np.exp(-alpha * x ** 2 / 2)
    return derv.squeeze()


def second_derivative(x):
    mass = pv.Constants.reduced_mass('O-H')
    omega = pv.Constants.convert(3700, 'wavenumbers', to_AU=True)
    alpha = mass * omega
    sderv = (alpha / np.pi) ** 0.25 * (alpha ** 2 * x ** 2 - alpha) * np.exp(-alpha * x ** 2 / 2)
    return sderv.squeeze()


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
