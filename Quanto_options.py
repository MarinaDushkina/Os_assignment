import pandas as pd
import numpy as np
from scipy.stats import norm
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, vega
from py_vollib.black_scholes_merton import implied_volatility as ivv

r = 0.03
fx_rate = 1.10
sigma_fx = 0.1
rho = 0.5

trade_data = pd.read_csv('trade_data.csv')
market_data = pd.read_csv('market_data.csv')
combined_data = pd.merge(trade_data, market_data, on='underlying', how='left')


def black_scholes_call(S, K, T, r, sigma_S):
    return black_scholes('c', S, K, T, r, sigma_S)


def black_scholes_put(S, K, T, r, sigma_S):
    return black_scholes('p', S, K, T, r, sigma_S)


def quanto_price(flag, S, K, T, r, rho, fx_rate, sigma_S, sigma_fx):
    d1 = (np.log(S / K) + (r - rho * sigma_S * sigma_fx + 0.5 * sigma_S ** 2) * T) / (
            sigma_S * np.sqrt(T))
    d2 = d1 - sigma_S * np.sqrt(T)
    if flag == 'c':
        quanto = fx_rate * (
                S * np.exp((- rho * sigma_S * sigma_fx) * T) * norm.cdf(d1) - K * np.exp(
            - r * T) * norm.cdf(d2))
    else:
        quanto = fx_rate * (K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(
            (- rho * sigma_S * sigma_fx) * T) * norm.cdf(-d1))

    return quanto


def calculate_implied_volatility(row):
    S = row['spot_price']
    K = row['strike']
    T = row['expiry']
    r = 0.03
    divs = 0.0
    market_price = row['PV'] / row['quantity']
    option_type = 'c' if row['call_put'] == 'CALL' else 'p'

    try:
        iv = ivv.implied_volatility(market_price, S, K, T, r, divs, option_type)
        iv = round(iv, 4)
    except Exception as e:
        iv = None
        # print(f"Error calculating implied volatility for row: {row}, Error: {e}")

    return iv


def calculate_metrics(row):
    S = row['spot_price']
    K = row['strike']
    T = row['expiry']
    sigma_S = row['volatility']
    quantity = row['quantity']
    option_type = 'c' if row['call_put'] == 'CALL' else 'p'

    if row['type'] == 'REGULAR':
        delta_ = quantity * delta(option_type, S, K, T, r, sigma_S)
        vega_ = quantity * vega(option_type, S, K, T, r, sigma_S)
        if row['call_put'] == 'CALL':
            pv = black_scholes_call(S, K, T, r, sigma_S)
        else:
            pv = black_scholes_put(S, K, T, r, sigma_S)

    if row['type'] == 'ODD':
        if row['call_put'] == 'CALL':
            pv = quanto_price('c', S, K, T, r, rho, fx_rate, sigma_S, sigma_fx)
            pv_delta = quanto_price('c', 1.01 * S, K, T, r, rho, fx_rate, sigma_S, sigma_fx)
            pv_vega = quanto_price('c', S, K, T, r, rho, fx_rate, sigma_S + 0.01, sigma_fx)
            delta_ = quantity * (pv_delta - pv) / (0.01 * S)
            vega_ = quantity * (pv_vega - pv) / 0.01
        else:
            pv = quanto_price('p', S, K, T, r, rho, fx_rate, sigma_S, sigma_fx)
            pv_delta = quanto_price('p', 1.01 * S, K, T, r, rho, fx_rate, sigma_S, sigma_fx)
            pv_vega = quanto_price('p', S, K, T, r, rho, fx_rate, sigma_S + 0.01, sigma_fx)
            delta_ = quantity * (pv_delta - pv) / (0.01 * S)
            vega_ = quantity * (pv_vega - pv) / 0.01

    pv *= row['quantity']

    return pd.Series([pv, delta_, vega_], index=['PV', 'Delta', 'Vega'])


combined_data[['PV', 'Delta', 'Vega']] = combined_data.apply(calculate_metrics, axis=1)
output_file_path = 'result.csv'
combined_data[['trade_id', 'PV', 'Delta', 'Vega']].to_csv(output_file_path, index=False)

regular_options = combined_data[combined_data['type'] == 'REGULAR'].copy()
regular_options.loc[:, 'Implied_Volatility'] = regular_options.apply(calculate_implied_volatility, axis=1)

output_data = regular_options[['underlying', 'Implied_Volatility']]
output_file_path = 'result_iv.csv'
output_data.to_csv(output_file_path, index=False)
