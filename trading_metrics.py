import numpy as np


def cumulative_return(P_end, P_0):
    """
    Calculate the Cumulative Return (CR).

    :param P_end: Final value of the portfolio
    :param P_0: Initial value of the portfolio
    :return: Cumulative Return
    """
    return (P_end - P_0) / P_0


def max_earning_rate(A_x_list, A_y_list):
    earning_rates = [(A_x - A_y) / A_y for A_x, A_y in zip(A_x_list, A_y_list) if A_y < A_x]
    return max(earning_rates) if earning_rates else 0


def maximum_pullback(A_x_list, A_y_list):
    pullbacks = [(A_x - A_y) / A_y for A_x, A_y in zip(A_x_list, A_y_list) if A_y > A_x]
    return max(pullbacks) if pullbacks else 0


def average_profitability_per_trade(P_end, P_0, NT):
    """
    Calculate the Average Profitability Per Trade (APPT).

    :param P_end: Returns at the end of trading stage
    :param P_0: Returns at the beginning of trading stage
    :param NT: Number of trades
    :return: Average Profitability Per Trade
    """
    return (P_end - P_0) / NT


def sharpe_ratio(E_Rp, R_f, sigma_P):
    """
    Calculate the Sharpe Ratio (SR).

    :param E_Rp: Expected return of the portfolio
    :param R_f: Risk-free rate
    :param sigma_P: Standard deviation of the portfolio's excess return
    :return: Sharpe Ratio
    """
    return (E_Rp - R_f) / sigma_P


def calculate_metrics(equity):
    # Calculate metrics
    P_end = equity.iloc[-1]
    P_0 = equity.iloc[0]
    NT = len(equity)  # Assuming one trade per time step
    E_Rp = equity.pct_change().mean()  # Expected return
    sigma_P = equity.pct_change().std()  # Standard deviation of returns

    # You'll need to adapt the below calculations depending on how your 'stats' are structured
    A_x = equity.values  # Assuming these are total assets over time
    A_y = np.roll(A_x, 1)  # Shifted assets for comparison
    A_y[0] = 1_000_000  # Initial asset value for the first calculation

    # Assuming 252 trading days in a year and 6.5 trading hours per day
    minutes_in_trading_year = 252 * 6.5 * 60
    # Your risk-free rate, annualized
    annual_risk_free_rate = 0.00
    # Convert the annual risk-free rate to a per-minute rate
    R_f = (1 + annual_risk_free_rate) ** (1 / minutes_in_trading_year) - 1
    # Annualize the expected return and standard deviation
    E_Rp = E_Rp * np.sqrt(minutes_in_trading_year)
    sigma_P = sigma_P * np.sqrt(minutes_in_trading_year)

    cr = cumulative_return(P_end, P_0)
    mer = max_earning_rate(A_x, A_y)
    mpb = maximum_pullback(A_x, A_y)
    appt = average_profitability_per_trade(P_end, P_0, NT)
    sr = sharpe_ratio(E_Rp, R_f, sigma_P)
    return cr, mer, mpb, appt, sr
