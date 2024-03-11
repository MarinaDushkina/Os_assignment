import pandas as pd
import numpy as np
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, vega
from py_vollib.black_scholes_merton import implied_volatility as ivv

r = 0.03
fx_rate = 1.10
sigma_fx = 0.1
rho = 0.5
n_simulations = 10000


trade_data = pd.read_csv('trade_data.csv')
market_data = pd.read_csv('market_data.csv')
combined_data = pd.merge(trade_data, market_data, on='underlying', how='left')


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
        #print(f"Error calculating implied volatility for row: {row}, Error: {e}")

    return iv


def simulate_fx_rate(S0, sigma, T, r, n_simulations):
    dt = 1/365
    n_steps = int(T/dt)
    random_walk = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, (n_simulations, n_steps)))
    S = S0 * random_walk.cumprod(axis=1)

    return S[:, -1]


def calculate_metrics(row):
    option_type = 'c' if row['call_put'] == 'CALL' else 'p'
    S = row['spot_price']
    K = row['strike']
    T = row['expiry']
    sigma = row['volatility']
    quantity = row['quantity']

    bs_price = black_scholes(option_type, S, K, T, r, sigma)
    bs_price *= quantity

    delta_ = delta(option_type, S, K, T, r, sigma)
    vega_ = vega(option_type, S, K, T, r, sigma)
    delta_total = delta_ * quantity
    vega_total = vega_ * quantity

    if row['type'] == 'ODD':
        simulated_fx_rates = simulate_fx_rate(fx_rate, sigma_fx, T, r, n_simulations)
        fx_adjusted_prices = bs_price / simulated_fx_rates
        option_price = np.mean(fx_adjusted_prices)
    else:
        option_price = bs_price

    return pd.Series([option_price, delta_total, vega_total], index=['PV', 'Delta', 'Vega'])


combined_data[['PV', 'Delta', 'Vega']] = combined_data.apply(calculate_metrics, axis=1)
output_file_path = 'result.csv'
combined_data[['trade_id', 'PV', 'Delta', 'Vega']].to_csv(output_file_path, index=False)

regular_options = combined_data[combined_data['type'] == 'REGULAR'].copy()
regular_options.loc[:, 'Implied_Volatility'] = regular_options.apply(calculate_implied_volatility, axis=1)

output_data = regular_options[['underlying', 'Implied_Volatility']]
output_file_path = 'result_iv.csv'
output_data.to_csv(output_file_path, index=False)