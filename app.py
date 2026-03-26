import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats

def get_bulletproof_analysis(fund_ticker, bench_ticker='VTI', window=126):
    # 1. Load Data
    df = yf.download([fund_ticker, bench_ticker], period='10y')['Adj Close']
    returns = df.pct_change().dropna()
    
    # 2. Rolling Regression (Beta & Alpha)
    # We calculate Beta to see how much of the return is just the market.
    rol_cov = returns[fund_ticker].rolling(window).cov(returns[bench_ticker])
    rol_var = returns[bench_ticker].rolling(window).var()
    rolling_beta = rol_cov / rol_var
    
    # Alpha = Fund Return - (Beta * Benchmark Return)
    # This is the "Skill" layer stripped of market influence.
    rolling_alpha = returns[fund_ticker].rolling(window).mean() - \
                    (rolling_beta * returns[bench_ticker].rolling(window).mean())
    
    # 3. Refined Z-Score: The "Skill Consistency" Metric
    # We calculate the Z-score of the Alpha relative to its own historical volatility.
    alpha_std = rolling_alpha.rolling(window).std()
    z_score_alpha = (rolling_alpha - rolling_alpha.rolling(window*2).mean()) / alpha_std
    
    # 4. Statistical "Checkmate" (The P-Value)
    # Is the outperformance statistically significant or random noise?
    t_stat = (rolling_alpha * np.sqrt(window)) / alpha_std
    p_values = 1 - stats.norm.cdf(t_stat) # Probability it was just luck

    # 5. Output for GitHub
    summary = pd.DataFrame({
        'Rolling_Beta': rolling_beta,
        'Alpha_ZScore': z_score_alpha,
        'Luck_Probability': p_values
    }).dropna()
    
    return summary

# To use for your post:
# stats = get_bulletproof_analysis('TRAIX')
# print(stats.tail())
