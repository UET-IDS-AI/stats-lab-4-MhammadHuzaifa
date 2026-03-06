"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    STEP 1
    Compute analytically

        P(X > 5)
        P(X < 5)
        P(3 < X < 7)

    STEP 2
    Simulate 100000 samples from Exp(1)

    STEP 3
    Estimate P(X > 5) using simulation
    """

    # Analytical (Exp(1))
    analytic_gt5 = math.exp(-5)
    analytic_lt5 = 1 - math.exp(-5)
    analytic_interval = math.exp(-3) - math.exp(-7)

    # Simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    return (
        analytic_gt5,
        analytic_lt5,
        analytic_interval,
        simulated_gt5,
    )


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF

        f(x) = 2x e^{-x^2} for x >= 0
    """

    # Define function
    def f(x):
        return 2 * x * np.exp(-x**2)

    # STEP 1: non-negativity check
    xs = np.linspace(0, 10, 1000)
    non_negative = np.all(f(xs) >= 0)

    # STEP 2: integral from 0 to infinity
    integral_value, _ = quad(lambda x: 2*x*np.exp(-x**2), 0, np.inf)

    # STEP 3: check if valid PDF
    is_valid_pdf = non_negative and abs(integral_value - 1) < 1e-3

    # STEP 4: plot
    x_plot = np.linspace(0, 3, 400)
    y_plot = f(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.title("PDF: f(x) = 2x e^{-x^2}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    return (
        integral_value,
        is_valid_pdf,
    )


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(1)
    """

    # Analytical
    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    # Simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    return (
        analytic_gt5,
        analytic_interval,
        simulated_gt5,
        simulated_interval,
    )


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10,2^2)
    """

    mu = 10
    sigma = 2

    # Analytical using normal CDF
    analytic_le12 = norm.cdf(12, mu, sigma)
    analytic_interval = norm.cdf(12, mu, sigma) - norm.cdf(8, mu, sigma)

    # Simulation
    samples = np.random.normal(mu, sigma, 100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    return (
        analytic_le12,
        analytic_interval,
        simulated_le12,
        simulated_interval,
    )
