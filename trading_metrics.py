def cumulative_return(P_end, P_0):
    """
    Calculate the Cumulative Return (CR).

    :param P_end: Final value of the portfolio
    :param P_0: Initial value of the portfolio
    :return: Cumulative Return
    """
    return (P_end - P_0) / P_0


def max_earning_rate(A_x_list, A_y_list):
    """
    Calculate the Maximum Earning Rate (MER).
    Here, A_x_list and A_y_list are lists of assets at different times, with the assumption
    that for each i, A_x_list[i] corresponds to a later time than A_y_list[i].

    :param A_x_list: List of total assets of the strategy at time x
    :param A_y_list: List of total assets of the strategy at time y, where y < x
    :return: Maximum Earning Rate
    """
    return max([(A_x - A_y) / A_y for A_x, A_y in zip(A_x_list, A_y_list)])


def maximum_pullback(A_x_list, A_y_list):
    """
    Calculate the Maximum Pullback (MPB).
    Here, A_x_list and A_y_list are lists of assets at different times, with the assumption
    that for each i, A_x_list[i] corresponds to a later time than A_y_list[i].

    :param A_x_list: List of total assets of the strategy at time x
    :param A_y_list: List of total assets of the strategy at time y, where y > x
    :return: Maximum Pullback
    """
    # In this case, we look for the maximum pullback, so we're interested in the instances where A_x < A_y
    pullbacks = [(A_y - A_x) / A_y for A_x, A_y in zip(A_x_list, A_y_list) if A_x < A_y]
    return max(pullbacks) if pullbacks else 0  # Return 0 if there are no pullbacks


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
