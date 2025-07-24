import warnings

import numpy as np
import scipy.special as sc


def calculate_epbeta_u_value(
    alpha: float|list[float],
    beta: float|list[float],
    epsilon_0: float|list[float],
    epsilon_1: float|list[float],
    cutpoint: float|list[float] = 0.5
    ) -> float|list[float]:
    """
    Calculate the u-function value for the EpBeta distribution given the parameters.

    Parameters:
    ----------
    alpha : float or list of floats
        The alpha parameter(s) of the EpBeta distribution. Must be non-negative.
    
    beta : float or list of floats
        The beta parameter(s) of the EpBeta distribution. Must be non-negative.
    
    epsilon_0 : float or list of floats
        The epsilon_0 parameter(s) of the EpBeta distribution. Must be non-negative.
    
    epsilon_1 : float or list of floats
        The epsilon_1 parameter(s) of the EpBeta distribution. Must be non-negative.
    
    cutpoint : float or list of floats, optional, default=0.5
        The cutpoint value(s) used in the calculation of the tilde_epsilon.

    Returns:
    -------
    float or list of floats
        The calculated u-function value(s) for the given parameters of the EpBeta distribution.
    """
    if (epsilon_0<0) or (epsilon_1<0):
        raise ValueError("'epsilon' must be non-negative.")
    if np.any(alpha<0) or np.any(beta<0):
        raise ValueError("'alpha' and 'beta' must be non-negative.")

    tilde_epsilon = (cutpoint+epsilon_0)/(1+epsilon_0+epsilon_1)

    return (
        ( (1+epsilon_0)*beta - epsilon_1*alpha ) / (alpha+beta)
            + 2 * (1+epsilon_0+epsilon_1) * sc.betainc(alpha+1, beta, tilde_epsilon) * alpha/(alpha+beta)
            - (1+2*epsilon_0) * sc.betainc(alpha, beta, tilde_epsilon)
        )


def get_epbeta_parameter(
    epsilon_0: float = 0.30,
    epsilon_1: float = 0.30,
    delta: float = 0.01,
    search_counts: int = 1_000_000,
    larger_alpha: bool|None = None,
    return_all_parameter: bool = False,
    tol: float = 1e-5
    ) -> tuple[float|np.ndarray]|None:
    """
    Compute the parameters for an EpBeta (or Beta) distribution based on the given parameters.

    Parameters:
    ----------
    epsilon_0 : float, default=0.30
        The epsilon_0 parameter for the EpBeta distribution. Must be non-negative.
    
    epsilon_1 : float, default=0.30
        The epsilon_1 parameter for the EpBeta distribution. Must be non-negative.
    
    delta : float, default=0.01
        The delta value used to filter the EpBeta distribution parameters. Must be non-negative.
    
    search_counts : int, optional, default=1_000_000
        The number of points to search for valid alpha and beta parameters.
    
    larger_alpha : bool, optional, default=True
        If True, ensures alpha is greater than or equal to beta. If False, ensures alpha is less than or equal to beta.
    
    return_all_parameter : bool, default=True
        If True, return np.ndarray with all possible parameters, ignoring `delta`.
    
    tol : float, optional, default=1e-5
        The tolerance level for comparing the computed u_value to delta.

    Returns:
    -------
    alpha : tuple or None
        The alpha parameter of the EpBeta or Beta distribution, or None if no valid parameter is found.
        If `return_all_parameter=True`, return np.ndarray with all possible parameters. 
    
    beta : tuple or None
        The beta parameter of the EpBeta or Beta distribution, or None if no valid parameter is found.
        If `return_all_parameter=True`, return np.ndarray with all possible parameters. 
    """
    # 0. checking input
    if (epsilon_0<0) or (epsilon_1<0):
        raise ValueError("'epsilon' must be non-negative.")
    if (delta<0):
        raise ValueError("'delta' must be non-negative.")

    # 1. get alpha beta
    if epsilon_0+epsilon_1 == 0:
        # 1-1. Beta distribution
        warnings.warn("Parameters for 'Beta' distribution are returned.", Warning)
        alpha = np.linspace(1e-10, 1, search_counts)
        beta = alpha.copy()
    else:
        # 1-2. EpBeta distribution
        a_b_slope = np.tan(np.linspace(1e-10, np.pi/2, search_counts))
        a_b_plus = (1+epsilon_0+epsilon_1)**2 / (1+epsilon_1-epsilon_0*a_b_slope) / (1+epsilon_0-epsilon_1/a_b_slope) - 1

        a_b_slope = a_b_slope[a_b_plus>0]
        a_b_plus = a_b_plus[a_b_plus>0]

        alpha = a_b_plus / (1+a_b_slope)
        beta = a_b_plus - alpha

        if larger_alpha is None:
            pass
        elif larger_alpha:
            indicator = alpha >= beta
            alpha = np.flip(alpha[indicator]) 
            beta = np.flip(beta[indicator])
        elif not larger_alpha:
            indicator = alpha <= beta
            alpha = alpha[indicator]
            beta = beta[indicator]

    # 2. compare delta
    u_value = calculate_epbeta_u_value(alpha, beta, epsilon_0, epsilon_1)

    indicator = u_value <= delta
    alpha, beta, u_value = alpha[indicator], beta[indicator], u_value[indicator]

    if (len(alpha) == 0):
        warnings.warn("There is no porper parameter.", Warning)
        return None, None
    elif return_all_parameter:
        return alpha, beta
    else:
        indicator = np.argmax(u_value)
        if (delta-u_value[indicator]) > tol:
            warnings.warn(f"Parameters for different delta value ({u_value[indicator]:0.5f}) are returned.", Warning)

        return alpha[indicator], beta[indicator]


def generate_epbeta_weight(
    size: int,
    epsilon_0: float,
    epsilon_1: float,
    alpha: float|None = None,
    beta: float|None = None,
    delta: float|None = None,
    seed: int|None = None,
    verbos: bool = False
    ) -> np.ndarray:
    """
    Generate random weights from an EpBeta distribution based on given parameters.

    Parameters:
    ----------
    size : int
        The number of random weights to generate.
    
    epsilon_0 : float
        The epsilon_0 parameter for the EpBeta distribution. Must be non-negative.
    
    epsilon_1 : float
        The epsilon_1 parameter for the EpBeta distribution. Must be non-negative.
    
    alpha : float or None, optional
        The alpha parameter of the EpBeta distribution. Must be non-negative. Ignored if delta is provided.
    
    beta : float or None, optional
        The beta parameter of the EpBeta distribution. Must be non-negative. Ignored if delta is provided.
    
    delta : float or None, optional
        The delta value used to derive parameters. If delta is provided, alpha and beta are ignored.
    
    seed : int or None, optional
        The seed for the random weight generator.

    verbos : bool, optional
        Print the parameter.

    Returns:
    -------
    np.ndarray
        An array of random weights generated from the EpBeta distribution.
    """
    if (epsilon_0<0) or (epsilon_1<0):
        raise ValueError("'epsilon' must be non-negative.")

    if delta is not None:
        if (alpha is not None) or (beta is not None):
            warnings.warn("When delta is provided, input parameters 'alpha' and 'beta' are ignored.", Warning)
        alpha, beta = get_epbeta_parameter(epsilon_0, epsilon_1, delta)
        if alpha is None:
            raise ValueError("There are no proper parameters for the given delta.")
    else:
        # delta is None
        delta = calculate_epbeta_u_value(alpha, beta, epsilon_0, epsilon_1)
        if (alpha is None) or (beta is None) or (alpha<0) or (beta<0):
            raise ValueError("'alpha' and 'beta' must be non-negative values.")

    if seed is not None:
        np.random.seed(seed)

    if verbos:
        if epsilon_0+epsilon_1 == 0:
            print(f"Generated from Beta({alpha:.4f},{beta:.4f}) where delta={delta:.4f}")
        else:
            print(f"Generated from EpBeta({alpha:.4f},{beta:.4f};{epsilon_0:.4f},{epsilon_1:.4f}) where delta={delta:.4f}")

    return np.random.beta(alpha, beta, size)*(1+epsilon_0+epsilon_1)-epsilon_0