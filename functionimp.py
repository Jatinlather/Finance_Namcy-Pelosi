import pandas as pd
import numpy as np
import random
from datetime import datetime

def getTickerPrice(ticker: str, date: datetime) -> float:
    # This is a stub implementation; replace with actual market data retrieval.
    return random.uniform(1, 100)

def calculate_trade_performance(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series({
            'Net Profit': 0,
            'roi':0,
            'Margin Equity': 0,
            'Sharpe Ratio': 0,
            'Sortino Ratio': 0,
            'Calmar Ratio': 0,
            'Profit Factor': 0,
            'Win Ratio': 0,
            'Average Winner': 0,
            'Average Loser': 0,
            'Max Drawdown': 0,
            'Risk-Reward Ratio': 0
            
        })

    trades['Size'] = trades['Size'].fillna(1)

    # Calculate P&L for each trade
    trades['PnL'] = trades.apply(
        lambda row: (getTickerPrice(row['Symbol'], pd.to_datetime(row['Date'])) - row['Price']) * row['Size'] 
        if row['Side'] == 'buy' else (row['Price'] - getTickerPrice(row['Symbol'], pd.to_datetime(row['Date']))) * row['Size'], 
        axis=1
    )

    # Net Profit
    net_profit = trades['PnL'].sum()

    #ROI
    initial_investment = trades['Price'].sum()
    roi = (net_profit / initial_investment) * 100

    # Margin Equity
    margin_equity = trades['PnL'].cumsum().iloc[-1]

    # Sharpe Ratio
    risk_free_rate = 0.01  # Assuming a 1% risk-free rate
    daily_returns = trades['PnL'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)

    # Sortino Ratio
    downside_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = (daily_returns.mean() - risk_free_rate) / downside_std * np.sqrt(252)

    # Calmar Ratio
    annual_return = daily_returns.mean() * 252
    max_drawdown = (trades['PnL'].cumsum().cummax() - trades['PnL'].cumsum()).max()
    calmar_ratio = annual_return / max_drawdown
    
    # Gross Profit and Gross Loss
    gross_profit = trades[trades['PnL'] > 0]['PnL'].sum()
    gross_loss = trades[trades['PnL'] < 0]['PnL'].sum()

    # Profit Factor
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf')
    
    # Win Ratio
    win_ratio = (trades['PnL'] > 0).mean()
    
    # Average Winner and Average Loser
    average_winner = trades[trades['PnL'] > 0]['PnL'].mean() if (trades['PnL'] > 0).any() else 0
    average_loser = trades[trades['PnL'] < 0]['PnL'].mean() if (trades['PnL'] < 0).any() else 0
    
    

    # Risk-Reward Ratio
    risk_reward_ratio = average_winner / abs(average_loser) if average_loser != 0 else float('inf')

    
    


    return pd.Series({
        'Net Profit': net_profit,
        'roi':roi,
        'Margin Equity': margin_equity,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Profit Factor': profit_factor,
        'Win Ratio': win_ratio,
        'Average Winner': average_winner,
        'Average Loser': average_loser,
        'Max Drawdown': max_drawdown,
        'Risk-Reward Ratio': risk_reward_ratio
    })

# Example usage
data = {
    'Date': [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
    'Symbol': ['AAPL', 'AAPL', 'GOOG'],
    'Side': ['buy', 'sell', 'buy'],
    'Size': [10, 5, 15],
    'Price': [150, 152, 1200]
}

trades_df = pd.DataFrame(data)
performance_metrics = calculate_trade_performance(trades_df)
print(performance_metrics)
