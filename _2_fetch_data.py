from _1_config import *
from _3_utils import *
from _4_feat_engineering import *
from _5_modify_data import *


def get_stock_data(symbol, start_date, end_date):
    return make_tz_naive(yf.Ticker(symbol).history(start=start_date, end=end_date))



def get_economic_indicators(fred, start_date, end_date):
    fred_series = {
        'GDP': 'GDP', 'Interest_Rates': 'FEDFUNDS', 'Consumer_Confidence': 'UMCSENT',
        'Industrial_Production': 'INDPRO', 'Unemployment_Rate': 'UNRATE',
        'Retail_Sales': 'RSAFS', 'Housing_Starts': 'HOUST', 'Corporate_Profits': 'CP',
        'Inflation_Rate': 'CPIAUCSL', 'Economic_Policy_Uncertainty': 'USEPUINDXD'
    }
    fred_data = pd.DataFrame({name: fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                              for name, series_id in fred_series.items()})
    return make_tz_naive(fred_data)

def get_market_indices(start_date, end_date):
    indices = yf.download(['^GSPC', '^VIX'], start=start_date, end=end_date)['Close']
    indices.columns = ['SP500', 'VIX']
    return make_tz_naive(indices)

def get_sector_data(symbol, start_date, end_date):
    sector_etfs = {
        'Information Technology': 'XLK', 'Health Care': 'XLV', 'Financials': 'XLF',
        'Consumer Discretionary': 'XLY', 'Communication Services': 'XLC',
        'Industrials': 'XLI', 'Consumer Staples': 'XLP', 'Energy': 'XLE',
        'Utilities': 'XLU', 'Real Estate': 'XLRE', 'Materials': 'XLB'
    }
    stock = yf.Ticker(symbol)
    sector = stock.info['sector']
    sector_etf = sector_etfs.get(sector, 'SPY')
    
    # Get the ETF price data
    sector_data = yf.download(sector_etf, start=start_date, end=end_date)['Close']
    
    # Create DataFrame with sector name as a constant column
    df = pd.DataFrame(index=sector_data.index)  # Use the same index as sector_data
    df['Sector'] = sector  # This will automatically broadcast to all rows
    df[f'{sector}_ETF'] = sector_data
    
    return make_tz_naive(df)

    ## Data Processing Functions

#! call in main, references other funcs
def collect_data(symbol, start_date, end_date, fred):
    print("\nData Collection")
    data_filename = f'stock_data_{symbol}_{start_date}_{end_date}.pkl'

    if not data_needs_update(os.path.join('stock_data', data_filename)):
        data, _ = load_data(os.path.join('stock_data', data_filename))
    else:
        stock_data = get_stock_data(symbol, start_date, end_date)
        eco_data = get_economic_indicators(fred, start_date, end_date)
        index_data = get_market_indices(start_date, end_date)
        sector_data = get_sector_data(symbol, start_date, end_date)
        data = pd.concat([stock_data, eco_data, index_data, sector_data], axis=1)
        save_data(data, data_filename)

    return data
