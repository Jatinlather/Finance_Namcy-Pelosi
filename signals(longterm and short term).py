import pandas as pd
import numpy as np
import random
from datetime import datetime

# Auxiliary function to get ticker price
def getTickerPrice(ticker: str, date: datetime) -> float:
    # This is a stub implementation; replace with actual market data retrieval.
    return random.uniform(1, 100)

# Function to preprocess the trades data
def preprocess_trades(df):
    # Mapping the type to Side
    df['Side'] = df['type'].apply(lambda x: 'buy' if 'Purchase' in x else 'sell')
    
    # Creating the Size column (assuming size as 1)
    df['Size'] = 1
    
    # Creating the Date column from transactionDate
    df['Date'] = pd.to_datetime(df['transactionDate'])
    
    # Mapping the ticker to Symbol
    df['Symbol'] = df['ticker']
    
    # Generating dummy prices for the example (normally, you would get these from market data)
    df['Price'] = df['ticker'].apply(lambda x: getTickerPrice(x, datetime.now()))
    
    # Selecting the required columns
    df = df[['Date', 'Symbol', 'Side', 'Size', 'Price']]
    
    return df

# Redefine the trade performance calculation function
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
    sharpe_ratio = (daily_returns.mean() - risk_free_rate) / daily_returns.std() * np.sqrt(252)#252 to normalize 

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



def moving_average_crossover(data, short_window=5, long_window=20):
    # Calculate moving averages
    data['Short_MA'] = data['Price'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Price'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate signals
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(
        data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, -1
    )
    data['Position'] = data['Signal'].diff()
    
    return data
# Load the testData.csv file
file_path = 'testData.csv'
trades_df = pd.read_csv(file_path)

# Preprocess the trades dataframe
processed_trades_df = preprocess_trades(trades_df)

# Apply the trade performance calculation function
performance_metrics = calculate_trade_performance(processed_trades_df)
print(performance_metrics)
# Apply the strategy
trades = moving_average_crossover(processed_trades_df)

# Extract buy and sell signals
buy_signals = trades[trades['Position'] == 1]
sell_signals = trades[trades['Position'] == -1]

print(buy_signals)
print(sell_signals)



