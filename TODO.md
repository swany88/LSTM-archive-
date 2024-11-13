get all stock market data for S&P 500 going back 20 years
save the data in "stock_data"
implement a check to see if the data is up to todays date
if up to date do not download new data
conduct correlation analysis on all stock close prices
predict on all stock close prices future values to time horizon 
find which one over time horizon has most predicted growth
implement connection to alpaca trade API to put in buy request on that stock 
on time horizon date put in sell order 

def get_sp500_symbols():
    """Get list of S&P 500 symbols using Wikipedia"""
    try:
        # Read S&P 500 table from Wikipedia
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error getting S&P 500 symbols: {e}")
        return []

def collect_all_sp500_data(start_date, end_date, fred):
    """Collect data for all S&P 500 stocks"""
    symbols = get_sp500_symbols()
    print(f"Found {len(symbols)} S&P 500 symbols")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(current_dir, 'stock_data'), exist_ok=True)
    
    # Get market and economic data once (shared across all stocks)
    eco_data = get_economic_indicators(fred, start_date, end_date)
    index_data = get_market_indices(start_date, end_date)
    
    failed_symbols = []
    for i, symbol in enumerate(symbols):
        try:
            print(f"\nProcessing {symbol} ({i+1}/{len(symbols)})")
            
            # Check if we already have recent data for this symbol
            data_filename = f'stock_data_{symbol}_{start_date}_{end_date}.pkl'
            filepath = os.path.join(current_dir, 'stock_data', data_filename)
            
            if not data_needs_update(filepath):
                print(f"Already have recent data for {symbol}")
                continue
                
            # Get stock specific data
            stock_data = get_stock_data(symbol, start_date, end_date)
            sector_data = get_sector_data(symbol, start_date, end_date)
            
            # Combine all data
            data = pd.concat([stock_data, eco_data, index_data, sector_data], axis=1)
            
            # Save to file
            save_data(data, data_filename)
            print(f"Saved data for {symbol}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            failed_symbols.append(symbol)
            continue
    
    if failed_symbols:
        print("\nFailed to collect data for these symbols:")
        print(", ".join(failed_symbols))
    
    return len(symbols) - len(failed_symbols)

# Update the main data collection section
if __name__ == "__main__":
    start_time = time.time()
    
    # Get data for last 10 years
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
    
    # Initialize FRED client
    fred_api_key = os.getenv('FRED_API_KEY')
    if not fred_api_key:
        raise ValueError("FRED_API_KEY not found in .env file")
    fred = Fred(api_key=fred_api_key)
    
    # Collect all S&P 500 data
    successful_collections = collect_all_sp500_data(start_date, end_date, fred)
    
    print(f"\nData collection completed in {(time.time() - start_time)/60:.2f} minutes")
    print(f"Successfully collected data for {successful_collections} stocks")