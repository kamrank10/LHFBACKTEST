import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Tuple, Optional
import pytz


class GapUpStrategyAnalyzer:
    def __init__(self, api_key: str):
        """
        Initialize the analyzer with Polygon.io API key

        Args:
            api_key: Your Polygon.io API key
        """
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.eastern = pytz.timezone('US/Eastern')

    def get_aggregates(self, ticker: str, date: str, timespan: str = "minute", multiplier: int = 1,
                       days_before: int = 0) -> Optional[Dict]:
        """
        Get aggregate bars for a specific ticker and date range

        Args:
            ticker: Stock ticker symbol
            date: End date in YYYY-MM-DD format
            timespan: minute, hour, day
            multiplier: Size of the timespan multiplier
            days_before: Number of days before the date to fetch
        """
        end_date = date
        if days_before > 0:
            start_date = (datetime.strptime(date, '%Y-%m-%d') - timedelta(days=days_before)).strftime('%Y-%m-%d')
        else:
            start_date = date

        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000
        }

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error fetching data for {ticker} on {date}: {response.status_code}")
                return None
        except Exception as e:
            print(f"Exception fetching data for {ticker}: {e}")
            return None

    def find_pivot_points(self, prices: pd.DataFrame, left_bars: int = 15, right_bars: int = 15) -> Tuple[
        List[float], List[float]]:
        """
        Find pivot highs and lows based on the TradingView logic

        Args:
            prices: DataFrame with 'high', 'low', 'close' columns
            left_bars: Number of bars to the left for pivot calculation
            right_bars: Number of bars to the right for pivot calculation

        Returns:
            Tuple of (resistance_levels, support_levels)
        """
        resistance_levels = []
        support_levels = []

        if len(prices) < left_bars + right_bars + 1:
            return resistance_levels, support_levels

        # Find pivot highs (resistance)
        for i in range(left_bars, len(prices) - right_bars):
            is_pivot_high = True
            pivot_value = prices.iloc[i]['high']

            # Check left bars
            for j in range(i - left_bars, i):
                if prices.iloc[j]['high'] >= pivot_value:
                    is_pivot_high = False
                    break

            # Check right bars
            if is_pivot_high:
                for j in range(i + 1, i + right_bars + 1):
                    if prices.iloc[j]['high'] >= pivot_value:
                        is_pivot_high = False
                        break

            if is_pivot_high:
                resistance_levels.append(pivot_value)

        # Find pivot lows (support)
        for i in range(left_bars, len(prices) - right_bars):
            is_pivot_low = True
            pivot_value = prices.iloc[i]['low']

            # Check left bars
            for j in range(i - left_bars, i):
                if prices.iloc[j]['low'] <= pivot_value:
                    is_pivot_low = False
                    break

            # Check right bars
            if is_pivot_low:
                for j in range(i + 1, i + right_bars + 1):
                    if prices.iloc[j]['low'] <= pivot_value:
                        is_pivot_low = False
                        break

            if is_pivot_low:
                support_levels.append(pivot_value)

        # Remove duplicates and sort
        resistance_levels = sorted(list(set(resistance_levels)), reverse=True)
        support_levels = sorted(list(set(support_levels)))

        return resistance_levels, support_levels

    def find_closest_resistances(self, current_price: float, resistance_levels: List[float], num_levels: int = 2) -> \
    List[Dict]:
        """
        Find the closest resistance levels above current price

        Args:
            current_price: Current price
            resistance_levels: List of resistance levels
            num_levels: Number of closest levels to return

        Returns:
            List of dictionaries with resistance info
        """
        above_price = [r for r in resistance_levels if r > current_price]
        above_price.sort()

        closest = []
        for i, level in enumerate(above_price[:num_levels]):
            closest.append({
                'level': level,
                'distance': level - current_price,
                'distance_pct': ((level - current_price) / current_price) * 100
            })

        return closest

    def get_support_resistance_for_next_day(self, ticker: str, gap_date: str, next_open: float) -> Dict:
        """
        Get support and resistance levels for the next trading day

        Args:
            ticker: Stock ticker symbol
            gap_date: Date of the gap up
            next_open: Opening price of next day

        Returns:
            Dictionary with support/resistance data
        """
        # Get daily data for the past 60 days to establish S/R levels
        data = self.get_aggregates(ticker, gap_date, "day", 1, days_before=60)

        if not data or 'results' not in data:
            return {}

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        if df.empty:
            return {}
            
        # Normalize column names for Polygon API response
        if 'h' in df.columns:
            df.rename(columns={'h': 'high', 'l': 'low', 'c': 'close'}, inplace=True)
        elif 'High' in df.columns:
            df.rename(columns={'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)

        # Find pivot points
        resistance_levels, support_levels = self.find_pivot_points(df, left_bars=10, right_bars=10)

        # Find closest resistances to next day's open
        closest_resistances = self.find_closest_resistances(next_open, resistance_levels, num_levels=2)

        # Find closest support below next open
        below_price = [s for s in support_levels if s < next_open]
        closest_support = below_price[-1] if below_price else None

        return {
            'all_resistances': resistance_levels[:5] if resistance_levels else [],
            'all_supports': support_levels[-5:] if support_levels else [],
            'closest_resistances': closest_resistances,
            'closest_support': closest_support,
            'resistance_1': closest_resistances[0]['level'] if len(closest_resistances) > 0 else None,
            'resistance_1_distance_pct': closest_resistances[0]['distance_pct'] if len(
                closest_resistances) > 0 else None,
            'resistance_2': closest_resistances[1]['level'] if len(closest_resistances) > 1 else None,
            'resistance_2_distance_pct': closest_resistances[1]['distance_pct'] if len(
                closest_resistances) > 1 else None,
        }

    def get_day_data_fast(self, ticker: str, gap_date: str) -> Dict:
        """
        Get HOD, closing price, total volume using 30-minute candles for speed

        Args:
            ticker: Stock ticker symbol
            gap_date: Date of the gap up

        Returns:
            Dictionary with gap day data including total volume
        """
        # Get 30-minute data for much faster processing
        data = self.get_aggregates(ticker, gap_date, "minute", 30)

        if not data or 'results' not in data:
            return {}

        bars = data['results']
        premarket_high = 0
        hod = 0
        hod_time = None
        close_price = 0
        total_volume = 0

        for bar in bars:
            bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.utc)
            bar_time_et = bar_time.astimezone(self.eastern)

            # Premarket (4:00 AM - 9:30 AM ET)
            if bar_time_et.hour < 9 or (bar_time_et.hour == 9 and bar_time_et.minute < 30):
                if bar_time_et.hour >= 4:
                    premarket_high = max(premarket_high, bar['h'])
            # Regular trading hours (9:30 AM - 4:00 PM ET)
            elif bar_time_et.hour < 16 or (bar_time_et.hour == 16 and bar_time_et.minute == 0):
                if bar['h'] > hod:
                    hod = bar['h']
                    hod_time = bar_time_et.strftime('%Y-%m-%d %H:%M:%S ET')
                # Track the last close price
                close_price = bar['c']
                # Add to total volume for regular trading hours
                total_volume += bar['v']

        return {
            'premarket_high': premarket_high if premarket_high > 0 else None,
            'hod': hod if hod > 0 else None,
            'hod_time': hod_time,
            'close': close_price if close_price > 0 else None,
            'total_volume': total_volume if total_volume > 0 else None,
            'hod_greater_than_pm': hod > premarket_high if premarket_high > 0 and hod > 0 else False
        }

    def get_next_day_data_fast(self, ticker: str, next_date: str) -> Dict:
        """
        Get next day's open, first 40 minutes high/low, and premarket volume using 30-minute candles

        Args:
            ticker: Stock ticker symbol
            next_date: Next trading day in YYYY-MM-DD format

        Returns:
            Dictionary with next day data including premarket volume (optimized)
        """
        # Get 30-minute data for much faster processing
        data = self.get_aggregates(ticker, next_date, "minute", 30)

        if not data or 'results' not in data:
            return {}

        bars = data['results']
        open_price = None
        first_40_high = 0
        first_40_low = float('inf')
        high_time = None
        low_time = None
        market_open_time = None
        premarket_volume = 0

        for bar in bars:
            bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.utc)
            bar_time_et = bar_time.astimezone(self.eastern)

            # Check for premarket volume (4:00 AM - 9:30 AM ET - full premarket session)
            if (bar_time_et.hour >= 4 and bar_time_et.hour < 9) or \
               (bar_time_et.hour == 9 and bar_time_et.minute < 30):
                premarket_volume += bar['v']
            
            # Check if it's during regular trading hours
            elif (bar_time_et.hour == 9 and bar_time_et.minute >= 30) or \
                    (bar_time_et.hour >= 10 and bar_time_et.hour < 16):

                # Get the open price (first RTH bar)
                if open_price is None:
                    open_price = bar['o']
                    market_open_time = bar_time_et

                # Check if within first 40 minutes (9:30 - 10:10)
                if market_open_time and (bar_time_et - market_open_time).total_seconds() <= 2400:  # 40 minutes
                    if bar['h'] > first_40_high:
                        first_40_high = bar['h']
                        high_time = bar_time_et.strftime('%Y-%m-%d %H:%M:%S ET')
                    if bar['l'] < first_40_low:
                        first_40_low = bar['l']
                        low_time = bar_time_et.strftime('%Y-%m-%d %H:%M:%S ET')

        return {
            'open': open_price,
            'first_40_high': first_40_high if first_40_high > 0 else None,
            'first_40_high_time': high_time,
            'first_40_low': first_40_low if first_40_low != float('inf') else None,
            'first_40_low_time': low_time,
            'premarket_volume': premarket_volume if premarket_volume > 0 else None
        }

    def get_volume_data_fast(self, ticker: str, gap_date: str, next_date: str) -> Dict:
        """
        Fast volume calculation using 30-minute candles - MUCH faster than minute data
        
        Args:
            ticker: Stock ticker symbol
            gap_date: Date of gap up
            next_date: Next trading day
            
        Returns:
            Dictionary with gap day volume and next day premarket volume
        """
        result = {
            'gap_day_volume': None,
            'next_day_premarket_volume': None
        }
        
        try:
            # Get gap day volume with 30-min candles (13 bars max per day vs 390 minute bars)
            gap_data = self.get_aggregates(ticker, gap_date, "minute", 30)
            if gap_data and 'results' in gap_data:
                gap_volume = 0
                for bar in gap_data['results']:
                    bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.utc)
                    bar_time_et = bar_time.astimezone(self.eastern)
                    
                    # Regular trading hours volume only
                    if (bar_time_et.hour == 9 and bar_time_et.minute >= 30) or \
                       (bar_time_et.hour >= 10 and bar_time_et.hour < 16):
                        gap_volume += bar['v']
                
                result['gap_day_volume'] = gap_volume if gap_volume > 0 else None
            
            # Get next day premarket volume with 30-min candles  
            next_data = self.get_aggregates(ticker, next_date, "minute", 30)
            if next_data and 'results' in next_data:
                premarket_volume = 0
                for bar in next_data['results']:
                    bar_time = datetime.fromtimestamp(bar['t'] / 1000, tz=pytz.utc)
                    bar_time_et = bar_time.astimezone(self.eastern)
                    
                    # Premarket hours (4:00 AM - 9:30 AM ET)
                    if (bar_time_et.hour >= 4 and bar_time_et.hour < 9) or \
                       (bar_time_et.hour == 9 and bar_time_et.minute < 30):
                        premarket_volume += bar['v']
                
                result['next_day_premarket_volume'] = premarket_volume if premarket_volume > 0 else None
                
        except Exception as e:
            print(f"    Volume calc error for {ticker}: {e}")
            
        return result

    def get_next_trading_day(self, current_date: str) -> str:
        """
        Get the next trading day after the given date

        Args:
            current_date: Date in YYYY-MM-DD format

        Returns:
            Next trading day in YYYY-MM-DD format
        """
        current = datetime.strptime(current_date, '%Y-%m-%d')
        next_day = current + timedelta(days=1)

        # Skip weekends
        while next_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            next_day += timedelta(days=1)

        return next_day.strftime('%Y-%m-%d')

    def format_price(self, price):
        """Helper function to safely format prices"""
        if price is None:
            return "N/A"
        return f"${price:.2f}"

    def analyze_ticker(self, ticker: str, gap_date: str, verbose: bool = False, debug: bool = False) -> Dict:
        """
        Analyze a single ticker that gapped up

        Args:
            ticker: Stock ticker symbol
            gap_date: Date when the stock gapped up
            verbose: If True, print detailed analysis

        Returns:
            Dictionary with all analysis data
        """
        print(f"\nAnalyzing {ticker} for gap on {gap_date}...")

        # Get gap day data
        gap_day_data = self.get_day_data(ticker, gap_date)

        # Only proceed if HOD > premarket high AND close < HOD
        hod_greater_than_pm = gap_day_data.get('hod_greater_than_pm', False)
        close_below_hod = False
        
        if gap_day_data.get('hod') and gap_day_data.get('close'):
            close_below_hod = gap_day_data['close'] < gap_day_data['hod']
        
        if not hod_greater_than_pm:
            print(f"  ❌ Skipping {ticker}: HOD not greater than premarket high")
            if verbose and gap_day_data.get('hod') and gap_day_data.get('premarket_high'):
                print(f"     HOD: ${gap_day_data['hod']:.2f}, PM High: ${gap_day_data['premarket_high']:.2f}")
            return None
            
        if not close_below_hod:
            print(f"  ❌ Skipping {ticker}: Close not below HOD")
            if verbose and gap_day_data.get('hod') and gap_day_data.get('close'):
                print(f"     HOD: ${gap_day_data['hod']:.2f}, Close: ${gap_day_data['close']:.2f}")
            return None

        # Get next trading day
        next_date = self.get_next_trading_day(gap_date)

        # Get next day data
        next_day_data = self.get_next_day_data(ticker, next_date)

        # Get support and resistance levels if we have next day open
        sr_data = {}
        if next_day_data.get('open'):
            sr_data = self.get_support_resistance_for_next_day(ticker, gap_date, next_day_data['open'])
        elif debug:
            print(f"  DEBUG: No next day open price found - cannot calculate resistance levels")

        result = {
            'ticker': ticker,
            'gap_date': gap_date,
            'premarket_high': gap_day_data.get('premarket_high'),
            'hod': gap_day_data.get('hod'),
            'hod_time': gap_day_data.get('hod_time'),
            'close': gap_day_data.get('close'),
            'next_date': next_date,
            'next_open': next_day_data.get('open'),
            'first_40_high': next_day_data.get('first_40_high'),
            'first_40_high_time': next_day_data.get('first_40_high_time'),
            'first_40_low': next_day_data.get('first_40_low'),
            'first_40_low_time': next_day_data.get('first_40_low_time'),
            'premarket_volume': next_day_data.get('premarket_volume'),
            'gap_day_volume': gap_day_data.get('total_volume'),
            # Add S/R data
            'resistance_1': sr_data.get('resistance_1'),
            'resistance_1_distance_pct': sr_data.get('resistance_1_distance_pct'),
            'resistance_2': sr_data.get('resistance_2'),
            'resistance_2_distance_pct': sr_data.get('resistance_2_distance_pct'),
            'closest_support': sr_data.get('closest_support')
        }

        # Print detailed analysis if verbose
        if verbose:
            self.print_detailed_analysis(result, sr_data)

        return result

    def print_detailed_analysis(self, result: Dict, sr_data: Dict):
        """Print detailed analysis for a ticker"""
        print("\n" + "=" * 60)
        print(f"DETAILED ANALYSIS FOR {result['ticker']}")
        print("=" * 60)

        print("\n📊 GAP DAY DATA:")
        print(f"  Date: {result['gap_date']}")
        print(f"  Premarket High: {self.format_price(result['premarket_high'])}")
        print(f"  High of Day: {self.format_price(result['hod'])} at {result['hod_time']}")
        print(f"  Close: {self.format_price(result['close'])}")

        if result['hod'] and result['premarket_high']:
            hod_pm_ratio = (result['hod'] / result['premarket_high'] - 1) * 100
            print(f"  HOD vs PM High: +{hod_pm_ratio:.2f}%")

        print("\n📈 NEXT DAY DATA:")
        print(f"  Date: {result['next_date']}")
        print(f"  Open: {self.format_price(result['next_open'])}")

        if result['next_open'] and result['close']:
            gap_pct = ((result['next_open'] - result['close']) / result['close']) * 100
            print(f"  Gap from previous close: {gap_pct:+.2f}%")

        print(f"\n  Volume Data:")
        if result['gap_day_volume']:
            print(f"    Gap Day Total: {result['gap_day_volume']:,} shares")
        else:
            print(f"    Gap Day Total: N/A")
            
        if result['premarket_volume']:
            print(f"    Next Day Premarket: {result['premarket_volume']:,} shares")
        else:
            print(f"    Next Day Premarket: N/A")
            
        print(f"\n  First 40 Minutes:")
        print(f"    High: {self.format_price(result['first_40_high'])} at {result['first_40_high_time']}")
        print(f"    Low: {self.format_price(result['first_40_low'])} at {result['first_40_low_time']}")

        if result['first_40_high'] and result['first_40_low'] and result['next_open']:
            range_40 = result['first_40_high'] - result['first_40_low']
            range_pct = (range_40 / result['next_open']) * 100
            print(f"    Range: ${range_40:.2f} ({range_pct:.2f}% of open)")

        print("\n🎯 RESISTANCE LEVELS:")
        if result['resistance_1']:
            print(
                f"  R1: {self.format_price(result['resistance_1'])} (+{result['resistance_1_distance_pct']:.2f}% from open)")

            # Check if R1 was hit in first 40 minutes
            if result['first_40_high'] and result['resistance_1'] <= result['first_40_high']:
                print(f"      ✅ R1 was hit in first 40 minutes")

                # Calculate potential profit
                if result['first_40_low']:
                    profit = ((result['resistance_1'] - result['first_40_low']) / result['resistance_1']) * 100
                    print(f"      💰 Potential profit (R1 to 40min low): {profit:.2f}%")
            else:
                print(f"      ❌ R1 was NOT hit in first 40 minutes")

        if result['resistance_2']:
            print(
                f"  R2: {self.format_price(result['resistance_2'])} (+{result['resistance_2_distance_pct']:.2f}% from open)")

            # Check if R2 was hit
            if result['first_40_high'] and result['resistance_2'] <= result['first_40_high']:
                print(f"      ✅ R2 was hit in first 40 minutes")

        if result['closest_support']:
            print(f"  Support: {self.format_price(result['closest_support'])}")

        # Show all resistance/support levels if available
        if sr_data.get('all_resistances'):
            print(f"\n  All Resistance Levels: {[f'${r:.2f}' for r in sr_data['all_resistances'][:5]]}")
        if sr_data.get('all_supports'):
            print(f"  All Support Levels: {[f'${s:.2f}' for s in sr_data['all_supports'][-5:]]}")

        print("\n" + "=" * 60)

    def test_single_ticker(self, ticker: str, gap_date: str, debug: bool = True):
        """
        Test the strategy on a single ticker with detailed output

        Args:
            ticker: Stock ticker symbol
            gap_date: Date when the stock gapped up (YYYY-MM-DD or MM/DD/YYYY)
            debug: If True, show debug information
        """
        # Convert date format if needed
        if '/' in gap_date:
            gap_date = datetime.strptime(gap_date, '%m/%d/%Y').strftime('%Y-%m-%d')

        print(f"\n{'=' * 60}")
        print(f"TESTING {ticker} on {gap_date}")
        print(f"{'=' * 60}")

        # Run with debug mode
        result = self.analyze_ticker(ticker.upper(), gap_date, verbose=True, debug=debug)

        if result:
            # Create a single-row DataFrame for consistency
            df = pd.DataFrame([result])

            # Save to a test file
            test_filename = f"test_{ticker}_{gap_date.replace('-', '_')}.csv"
            df.to_csv(test_filename, index=False)
            print(f"\n✅ Test results saved to {test_filename}")

            return df
        else:
            print(f"\n❌ {ticker} did not meet the criteria (HOD > PM High)")
            return None

    def analyze_all_tickers(self, input_file: str, output_file: str = 'gap_analysis_results.csv',
                            test_mode: bool = False, test_limit: int = 5):
        """
        Analyze all tickers from the input CSV file

        Args:
            input_file: Path to input CSV with tickers and gap dates
            output_file: Path to save results CSV
            test_mode: If True, only process first test_limit tickers
            test_limit: Number of tickers to process in test mode
        """
        # Read input CSV
        df = pd.read_csv(input_file)

        # Ensure proper column names
        if 'symbol' in df.columns:
            df.rename(columns={'symbol': 'ticker'}, inplace=True)
        if 'date' in df.columns and 'gap_date' not in df.columns:
            df.rename(columns={'date': 'gap_date'}, inplace=True)

        # Apply test limit if in test mode
        if test_mode:
            df = df.head(test_limit)
            print(f"\n🧪 TEST MODE: Processing only first {test_limit} tickers")

        print(f"Found {len(df)} tickers to analyze")
        print(f"Columns in input file: {df.columns.tolist()}")

        results = []
        successful_tickers = []
        failed_tickers = []

        for idx, row in df.iterrows():
            try:
                ticker = row['ticker']
                gap_date = row['gap_date']

                # Convert date to YYYY-MM-DD format if needed
                if pd.notna(gap_date) and '/' in str(gap_date):
                    gap_date = datetime.strptime(gap_date, '%m/%d/%Y').strftime('%Y-%m-%d')

                result = self.analyze_ticker(ticker, gap_date)

                if result:
                    results.append(result)
                    successful_tickers.append(ticker)

                    # Safe printing with None handling
                    pm_high = self.format_price(result['premarket_high'])
                    hod = self.format_price(result['hod'])
                    next_open = self.format_price(result['next_open'])
                    high_40 = self.format_price(result['first_40_high'])
                    low_40 = self.format_price(result['first_40_low'])
                    r1 = self.format_price(result.get('resistance_1'))
                    r2 = self.format_price(result.get('resistance_2'))

                    print(f"  ✓ {ticker}: HOD {hod} > PM High {pm_high}, Close {self.format_price(result['close'])}")
                    
                    pm_vol = f"{result['premarket_volume_30min']:,}" if result['premarket_volume_30min'] else "N/A"
                    print(f"    Next day: Open {next_open}, PM Vol: {pm_vol}")
                    print(f"    40min High {high_40}, 40min Low {low_40}")

                    # Print resistance levels if found
                    if result.get('resistance_1'):
                        r1_dist = result.get('resistance_1_distance_pct', 0)
                        print(f"    R1: {r1} (+{r1_dist:.2f}%)")
                    if result.get('resistance_2'):
                        r2_dist = result.get('resistance_2_distance_pct', 0)
                        print(f"    R2: {r2} (+{r2_dist:.2f}%)")
                else:
                    failed_tickers.append(ticker)

                # Save periodically (every 10 tickers)
                if (idx + 1) % 10 == 0 and not test_mode:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(f'temp_{output_file}', index=False)
                    print(f"Saved temporary results ({len(results)} tickers processed)")

            except Exception as e:
                print(f"Error processing {ticker} on {gap_date}: {e}")
                failed_tickers.append(ticker)
                continue

        # Save final results
        if results:
            results_df = pd.DataFrame(results)

            # Use different filename for test mode
            if test_mode:
                output_file = f"test_{output_file}"

            results_df.to_csv(output_file, index=False)
            print(f"\n{'=' * 60}")
            print(f"Analysis complete! Results saved to {output_file}")
            print(f"Total tickers analyzed: {len(results_df)}")
            print(f"Successful: {len(successful_tickers)}")
            print(f"Failed criteria: {len(failed_tickers)}")

            # Summary statistics
            self.print_summary_statistics(results_df)

            return results_df
        else:
            print("\nNo tickers met the criteria")
            return pd.DataFrame()

    def get_processed_tickers(self, temp_file: str) -> set:
        """Get list of already processed tickers from temp file to resume processing"""
        try:
            if pd and os.path.exists(temp_file):
                temp_df = pd.read_csv(temp_file)
                return set(temp_df['ticker'].tolist())
        except:
            pass
        return set()

    def analyze_resistance_levels_from_hod_file(self, hod_file: str, output_file: str = 'resistance_levels_analysis.csv', resume: bool = True):
        """
        Analyze resistance levels for all tickers from HOD Tickers CSV file
        
        Args:
            hod_file: Path to HOD Tickers CSV file  
            output_file: Output file name for resistance analysis results
        """
        print(f"\n{'=' * 60}")
        print("ANALYZING RESISTANCE LEVELS FROM HOD TICKERS")
        print(f"{'=' * 60}")
        
        # Read the HOD tickers file
        try:
            df = pd.read_csv(hod_file)
            print(f"Found {len(df)} total tickers in {hod_file}")
            
            # Resume from where we left off if requested
            if resume:
                temp_file = f'temp_{output_file}'
                processed_tickers = self.get_processed_tickers(temp_file)
                if processed_tickers:
                    df = df[~df['ticker'].isin(processed_tickers)]
                    print(f"Resuming: {len(processed_tickers)} already processed, {len(df)} remaining")
            
            print(f"Processing {len(df)} tickers")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading {hod_file}: {e}")
            return None
            
        results = []
        processed = 0
        errors = 0
        
        for idx, row in df.iterrows():
            try:
                ticker = row['ticker']
                next_date = row['next_date'] 
                next_open = row.get('next_open')
                
                if pd.isna(next_open) or next_open is None:
                    print(f"  ! Skipping {ticker}: No next day open price")
                    errors += 1
                    continue
                
                print(f"\n  Analyzing {ticker} for {next_date}...")
                
                # Calculate resistance levels using the next day's open price
                sr_data = self.get_support_resistance_for_next_day(ticker, row['gap_date'], next_open)
                
                # Get volume data FAST using 30-minute candles (2 API calls vs 4+)
                volume_data = self.get_volume_data_fast(ticker, row['gap_date'], next_date)
                
                # Create comprehensive result with all data including volume
                result = {
                    'ticker': ticker,
                    'gap_date': row['gap_date'],
                    'next_date': next_date,
                    'next_open': next_open,
                    'premarket_high': row.get('premarket_high'),
                    'hod': row.get('hod'),
                    'close': row.get('close'),
                    'first_40_high': row.get('first_40_high'),
                    'first_40_low': row.get('first_40_low'),
                    
                    # Volume data (FAST)
                    'gap_day_volume': volume_data.get('gap_day_volume'),
                    'next_day_premarket_volume': volume_data.get('next_day_premarket_volume'),
                    
                    # Resistance levels
                    'resistance_1': sr_data.get('resistance_1'),
                    'resistance_1_distance_pct': sr_data.get('resistance_1_distance_pct'),
                    'resistance_2': sr_data.get('resistance_2'),  
                    'resistance_2_distance_pct': sr_data.get('resistance_2_distance_pct'),
                    'closest_support': sr_data.get('closest_support'),
                    
                    # All resistance and support levels found
                    'all_resistances': str(sr_data.get('all_resistances', [])),
                    'all_supports': str(sr_data.get('all_supports', [])),
                    
                    # Analysis flags
                    'r1_hit_in_40min': sr_data.get('resistance_1') and row.get('first_40_high') and sr_data.get('resistance_1') <= row.get('first_40_high', 0),
                    'r2_hit_in_40min': sr_data.get('resistance_2') and row.get('first_40_high') and sr_data.get('resistance_2') <= row.get('first_40_high', 0)
                }
                
                results.append(result)
                processed += 1
                
                # Print key resistance info
                if sr_data.get('resistance_1'):
                    r1_dist = sr_data.get('resistance_1_distance_pct', 0)
                    r1_hit = "[HIT]" if result['r1_hit_in_40min'] else "[NOT HIT]"
                    print(f"    R1: ${sr_data['resistance_1']:.4f} (+{r1_dist:.2f}%) - {r1_hit}")
                if sr_data.get('resistance_2'):
                    r2_dist = sr_data.get('resistance_2_distance_pct', 0) 
                    r2_hit = "[HIT]" if result['r2_hit_in_40min'] else "[NOT HIT]"
                    print(f"    R2: ${sr_data['resistance_2']:.4f} (+{r2_dist:.2f}%) - {r2_hit}")
                    
                if sr_data.get('all_resistances'):
                    print(f"    All Resistances: {[f'${r:.4f}' for r in sr_data['all_resistances'][:5]]}")
                    
                # Save periodically every 20 tickers
                if (idx + 1) % 20 == 0:
                    temp_df = pd.DataFrame(results)
                    temp_df.to_csv(f'temp_{output_file}', index=False)
                    print(f"\n  [SAVE] Saved temporary results ({processed} tickers processed)")
                    
            except Exception as e:
                print(f"  X Error processing {ticker}: {e}")
                errors += 1
                continue
        
        # Save final results  
        if results:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            
            print(f"\n{'=' * 60}")
            print(f"[COMPLETE] RESISTANCE ANALYSIS COMPLETE!")
            print(f"{'=' * 60}")
            print(f"Total tickers processed: {processed}")
            print(f"Errors encountered: {errors}")
            print(f"Results saved to: {output_file}")
            
            # Quick statistics
            with_r1 = len(results_df.dropna(subset=['resistance_1']))
            with_r2 = len(results_df.dropna(subset=['resistance_2']))
            r1_hits = results_df['r1_hit_in_40min'].sum() if 'r1_hit_in_40min' in results_df.columns else 0
            r2_hits = results_df['r2_hit_in_40min'].sum() if 'r2_hit_in_40min' in results_df.columns else 0
            
            print(f"\nQUICK STATS:")
            print(f"   Tickers with R1: {with_r1}/{processed} ({with_r1/processed*100:.1f}%)")
            print(f"   Tickers with R2: {with_r2}/{processed} ({with_r2/processed*100:.1f}%)")
            print(f"   R1 hit in 40min: {r1_hits}/{with_r1} ({r1_hits/with_r1*100:.1f}%)" if with_r1 > 0 else "   R1 hit in 40min: N/A")
            print(f"   R2 hit in 40min: {r2_hits}/{with_r2} ({r2_hits/with_r2*100:.1f}%)" if with_r2 > 0 else "   R2 hit in 40min: N/A")
            
            return results_df
        else:
            print(f"\n[ERROR] No results to save")
            return pd.DataFrame()

    def print_summary_statistics(self, results_df: pd.DataFrame):
        """Print summary statistics for the analysis"""
        if len(results_df) == 0:
            return

        print(f"\n{'=' * 60}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 60}")

        # Filter out None values for calculations
        valid_ratios = results_df.dropna(subset=['hod', 'premarket_high'])
        if len(valid_ratios) > 0:
            avg_ratio = (valid_ratios['hod'] / valid_ratios['premarket_high']).mean()
            print(f"Average HOD/PM High ratio: {avg_ratio:.2%}")

        valid_gaps = results_df.dropna(subset=['next_open', 'close'])
        if len(valid_gaps) > 0:
            avg_gap = ((valid_gaps['next_open'] - valid_gaps['close']) / valid_gaps['close']).mean()
            print(f"Average next day gap: {avg_gap:.2%}")

        # Resistance statistics
        valid_r1 = results_df.dropna(subset=['resistance_1_distance_pct'])
        if len(valid_r1) > 0:
            avg_r1_dist = valid_r1['resistance_1_distance_pct'].mean()
            print(f"Average distance to R1: {avg_r1_dist:.2f}%")

            # Count how many hit R1
            r1_hit = results_df.dropna(subset=['resistance_1', 'first_40_high'])
            r1_hit_count = len(r1_hit[r1_hit['first_40_high'] >= r1_hit['resistance_1']])
            print(f"R1 hit in first 40 min: {r1_hit_count}/{len(r1_hit)} ({r1_hit_count / len(r1_hit) * 100:.1f}%)")

        valid_r2 = results_df.dropna(subset=['resistance_2_distance_pct'])
        if len(valid_r2) > 0:
            avg_r2_dist = valid_r2['resistance_2_distance_pct'].mean()
            print(f"Average distance to R2: {avg_r2_dist:.2f}%")

        # Potential profit analysis
        print(f"\n{'=' * 60}")
        print("POTENTIAL SHORT OPPORTUNITIES")
        print(f"{'=' * 60}")

        # Filter for good short setups
        short_setups = results_df.dropna(subset=['resistance_1_distance_pct', 'resistance_1', 'first_40_low'])
        short_setups = short_setups[short_setups['resistance_1_distance_pct'] < 2.0]  # R1 within 2% of open

        if len(short_setups) > 0:
            print(f"Found {len(short_setups)} tickers with R1 within 2% of next day open")

            # Calculate potential profit
            short_setups['potential_profit_pct'] = ((short_setups['resistance_1'] - short_setups['first_40_low']) /
                                                    short_setups['resistance_1']) * 100
            avg_profit = short_setups['potential_profit_pct'].mean()
            print(f"Average potential profit (R1 to 40min low): {avg_profit:.2f}%")

            # Show top opportunities
            top_opps = short_setups.nlargest(5, 'potential_profit_pct')[
                ['ticker', 'gap_date', 'resistance_1', 'first_40_low', 'potential_profit_pct']]
            print("\nTop 5 Short Opportunities:")
            for _, row in top_opps.iterrows():
                print(
                    f"  {row['ticker']}: R1=${row['resistance_1']:.2f} → Low=${row['first_40_low']:.2f} = {row['potential_profit_pct']:.2f}% profit")


# Main execution
if __name__ == "__main__":
    # Configuration
    API_KEY = "xyHWUl9BQ4VH9s5CkdZ7ZgsMFGZJzmLj"  # ← REPLACE THIS WITH YOUR ACTUAL API KEY
    # Example: API_KEY = "AbCdEfGhIjKlMnOpQrStUvWxYz123456"

    INPUT_FILE = "input.csv"  # Your input CSV file with tickers
    OUTPUT_FILE = "gap_strategy_results.csv"  # Output file name

    # Initialize analyzer
    analyzer = GapUpStrategyAnalyzer(API_KEY)

    # ==============================================================
    # NEW: ANALYZE RESISTANCE LEVELS FROM HOD TICKERS FILE
    # ==============================================================
    # Analyze resistance levels for all tickers in your HOD Tickers.csv file
    
    HOD_FILE = "HOD Tickers.csv"  # Your HOD tickers file
    RESISTANCE_OUTPUT = "resistance_levels_analysis.csv"  # Output file for resistance analysis
    
    # FAST analysis with optimized 30-minute candles and resume capability
    print("Starting FAST resistance level analysis with 30-min candles (MUCH faster!)...")
    VOLUME_OUTPUT = "resistance_levels_with_volume.csv"  # New output file with volume data
    
    # Resume from where we left off + use 30-min candles for speed
    results = analyzer.analyze_resistance_levels_from_hod_file(HOD_FILE, VOLUME_OUTPUT, resume=True)
    
    # ==============================================================
    # OPTION 1: TEST A SINGLE TICKER (COMMENTED OUT)
    # ==============================================================
    # Uncomment the lines below to test a single ticker:
    
    # test_ticker = "KAVL"  # Replace with your test ticker
    # test_date = "2024-01-15"  # Replace with the gap date (YYYY-MM-DD or MM/DD/YYYY)
    # analyzer.test_single_ticker(test_ticker, test_date)

    # ==============================================================
    # OPTION 2: TEST MODE - Process first few tickers from the list (COMMENTED OUT)
    # ==============================================================
    # Uncomment the line below to run in test mode (processes first 5 tickers):

    # results = analyzer.analyze_all_tickers(INPUT_FILE, OUTPUT_FILE, test_mode=True, test_limit=5)

    # ==============================================================
    # OPTION 3: FULL ANALYSIS - Process all tickers (COMMENTED OUT)
    # ==============================================================
    # Comment out the line below if running test mode:

    # results = analyzer.analyze_all_tickers(INPUT_FILE, OUTPUT_FILE)

    # ==============================================================
    # OPTION 4: CREATE INPUT CSV FROM YOUR TICKER LIST
    # ==============================================================
    # This section creates an input CSV from the tickers you provided
    # You'll need to add the gap dates for each ticker

    create_custom_csv = False  # Set to True to create the CSV

    if create_custom_csv:
        # Your complete ticker list from the document
        tickers_str = """VS AZTR OXBR ALZN CHNR KAVL HUSA MBIO LCFY CTNT NAOV RBNE SGN SGN MDRR TXMD NRSN ARTL CYCN LITM 
        CNEY FEMY AKAN SNTG XBIO WATT CLIK SSY APM LXEH UPC WORX IBO RELI TIRX RGS AUUD SPRC ITRM NVFY BTBD PRFX SQFT 
        APM GPUS MGIH CYCC EVOK ENSC MMA LRHC INM BGLC SCKT WLDS BMR FOXO HCTI MGRX SNTI RSLS PRTG JSPR SLRX BBLG KTTA 
        SLRX SOAR HCWB KXIN TPST PRTG MSAI IMRN OSRH VVOS PMN IGC KTTAW THAR AMBO PTIX NVVE LUCY VRAX UOKA TOP ENVB 
        CLPS VVOS CERO TRAW POLA TWG LASE ABVC ADN AMIX QLGN CHEK USEA COHN GTBP KAVL POAI NUWE AMST ADIL SYPR PHIO 
        AMIX ARBB ANTE GSUN AREB INAB AIFF CELZ DRCT PMCB SMXT BKYI KTTA JDZG SILO KXIN MIRA LIVE EJH SXTC MURA NVVE 
        OST REBN LUCY NIVF APLM PSIG HOTH GCTK BNZI TANH BNZI CWD ATCH AAME SABS BGLC HUSA FOXO PALI MARPS SVRE VMAR 
        OTRK TNON AGMH GOVX NLSP CLRB SXTP LIDRW SNOA PWM YJ MARPS CELZ FTFT NITO SINT IINN ACON SXTP OLB WORX GNPX 
        AREB NDRA VERB KZIA TIVC INTZ MTNB INM GSIW ENVB HOTH AMOD RMSG RIME EZGO AIRI TENX JSPR SGBX SGBX VTVT ZCMD 
        AMOD BAOS BFRG DTCK HOTH PPBT MRKR CANF GLMD NRSN LUCY EFOI SOND VERO IBG ALUR ONCO UAVS MIGI MOGO GTEC VEEE 
        HTCR AMST PALI VRAX SNGX CXAI VERB MGRX XELB ENVB DRCT ADIL NVVE OGEN ADXN SCNI KXIN SUNE OBLG QMMM SNES BENF 
        SINT MTC OMEX ENTO MNTS SNTG NVVE DUO HTOO WINT LGVN GPUS HOTH BON SPCB DPRO GFAI GMM UPC BNAI CISO VVOS OTRK 
        WIMI ELWS BOF SABS NAOV OLB SILO AIRE XRTX SNTI SXTP GLMD ALF SGRP BOF ANGH OTRK PLAG PPSI CANF MTC CAPT UOKA 
        AIRE PCSA PSTV WORX BIAF DWTX TGL MSGM GELS OBLG SGN DXF KZIA OCTO PLRZ BBLG AIMD ATCH SLRX NRSN AMOD WBUY 
        SYPR LGVN MITQ SMXT ATHE RNAZ VVPR KITT CLWT CREV PALI LUCY HUBC UUU ISPC ILAG GWH CASI FARM INBS MTEK GXAI 
        VIVK GTEC BKYI LEXX SYTAW SOBR LUCY CELZ COCP STI SBFM HOTH BEAT SIDU STSS TKLF ITRM SQFT SPPL PAPL WORX WHLR 
        IDAI KAPA FTFT APWC ATXG HTCR BIYA VS AUUD FCUV CDIO QLGN CPOP CAPS RLYB LGCL IFBD GFAI JSPRW GTEC KLTO ABVE 
        TPET SLNH NISN BMRA SLXN WLGS LMFA BRTX DXF NUKK RSLS STAK EDSA ATNF BCG NCRA CING LIXT NVVE RDHL KLTO AMBO 
        SNTI DATS GFAI DWTX RVSN FAAS DEVS ADIL AMST WORX INAB CERO GAME IPDN KOSS COSM ENSC CMND PCSA BGLC MCTR XCUR 
        MTC UAVS MNTS QMCO XYLO ADXN QNRX RELI AIMD MTC PRZO CPOP SLXN DXF USEG LOBO INTZ PEPG ABVC BAOS RENT FRGT 
        HUSA SWVL DTSS ARTL EDSA ENVB VS SLXN BLIN PRSO MNTS MNTS MNDR GXAI APM RANI PHIO OSRH SONN BIAF ACRV VSEE 
        LASE YOSH OP CHEK SJ LGVN KPRX LMFA CVM TGL NERV RVPH ADN COCH ARBEW NTRBW VRAX OPTT WAFU VSME RDHL ZOOZ JCSE 
        ATNFW SCKT KOSS GP WIMI SNGX CETX NBY LMFA ITRM YHC DYAI CNET AQMS AUUD SLXN IRIX SOPA TOPS APWC MTC AEMD 
        PRZO IPWR OBLG EONR QH BFRG SSY ABVC ATER ISPC DATS BJDX SGN JTAI SOPA ZKIN CXAI FRGT LXEH ILAG SYTA BBLG 
        CMND MRM SYTA LASE LCFY CREV ALF OSRH APRE GFAI ITP AERT TOP SIDU LRE RMTI TRIB BCTX ZOOZ FOXO SXTC CDIO MNTS 
        CLRB WAFU HUBC HOOK INDO DRMA RBNE VRAR MTC WBUY OMH KSCP DTSS SNTI DTSS JYD MTC LEXX INM JXG GWH MHUA XPON ADN 
        AGH LRHC PRFX CISO SGD LGVN LUCY JBDI ORIS CELZ ATXG LEDS GFAI DATS WIMI NVFY STAI BNAI FEMY VYNE KZR SYTA SAIH 
        SNTG WORX WORX BFRI GTI ISPOW CTXR SGBX EDBL MSGM KOSS ARTL RVSN ENSC ENGS PTNM CNET CGTX FTEL DTSS SGD MOGU 
        RNAZ GMM FOXO KAVL WCT LEDS BIAF OXBR JSPR DWTX SLRX IVDA RELI WORX GSUN MODV CETY HAO BLIN DATS ELWS MNTS LASE 
        KOSS VIVK PPSI BNRG APM ANY CLPS PETZ BFRI CLSD BEAT KAVL BNRG INDO GRNQ PWM BOF CLGN OTRK NITO LYRA RVSN SXTC 
        LEDS LIXT SEED TTNP BJDX MODV TNON BNRG ITP PAPL XOS HUSA SNTI FRSX LGVN AUUD RVSN LITM OXBR SEED WORX LITM 
        SOPA SIDU BTAI VS VERO SNTI ENLV RAIN PLRZ HUSA SNTG MOB PALI CPOP SPCB IMTE INDP SIDU SKK JTAI IINN GRNQ MNTS 
        AMIX KLTO TPET OLB CXAI WKHS INBS BFRI GDHG BTAI INAB GFAI DEVS NMTC GOVX EEIQ GIBO BANL MSN CTRM WVVI HWH NNVC 
        CLSD CARV ANY CYCN UUU LGVN NRXP IOTR LASE NTWK MYSZ SLRX TOVX STI SLRX WAFU SUGP BGFV SONM CISS DRCT DXYZ MTC 
        ITP TXMD MBIO OTRK KAVL TIRX MDRR RNAZ LASE PTIX TOP NAKA AEMD BTCM NCPL DLPN SONM NERV EKSO INM RNXT HTCR OXBR 
        DTST JFU JCSE PWM TOP IHT WIMI EFOI BOF NAAS ROLR IPW EDSA IGC ATER GSIW COHN POLA SIDU GTI JCSE GAME ATNF BOF 
        PSTV HKPD MSGM BKYI VVOS DATS AIRI MTEK ASTC PRPO DRCT GFAI CREV WIMI VTVT ABVE BCDA USEA MWG SIF DTSS FRSX PRZO 
        RCON LITM MNDR CXAI HOLO GP CREX POAI SPCB OGEN DLPN AMST DTCK VSME NXL MEGL NUKK TOP GAME AMIX PSTV NDRA LXEH 
        DXYZ PRZO EONR NXL SONM HUDI EDTK SOBR SXTC MIRA EPWK ABVE ATNF MITQ MRKR SYTAW TPIC TANH ALF OBLG DTSS SYBX 
        SOHO ICON KITT GRI ELWS SMSI ARQQW VYNE SPCB RMTI GFAI WKHS PCLA SOND GBR MOGU TXMD CREG FRSX ALF KPTI TXMD IXHL 
        SSY WAFU XTLB AEON LGHL EBON WNW CYN ONFO GV SGBX PSTV PW DTST CYN VRAR CLWT REBN SLNH AUST TRNR NVFY DATS PT 
        RIME MIGI IVP LMFA GNPX CLNN SOBR AIMD TPST HOTH SGBX SNTG ONMD MEGL INDO PLAG FEMY SOAR MODV ONMD NXL CMMB MRSN 
        VSME OPAD PT TCRT MYNZ SOPA LGVN VIVK PW PPSI FRSX JOB CETX TNFA SNOA XTLB HOTH RMTI PSHG LASE TDACW IGC GTI ITP 
        INDO ANY ATXG SBFM WLGS GOVX MRKR QMCO IPW GRNQ CPSH ISPOW IBIO ABVC TPST TIRX LXEH NCPL ATNF FOXO PHIO COSM 
        PLRZ PTLE IMTE KSCP SUUN ECDA MRM EZGO VMAR ADNWW LIVE MSPR UUU WNW VRAR WIMI KNW LOBO SNES DSS XRTX EPOW BURU 
        BJDX FRSX SNTG GBR XRTX TKLF LUCY INBS LASE CELZ EEIQ PALI MITQ SPHL NXL DPRO HUDI EPOW PRPO SQNS BTAI XCUR BMR 
        VRAX CDIO TPET ADTX ENVB GELS NCNA PHIO HYPD IMCC USEG HUDI UPC TNON YOSH HUSA OMEX SMSI PT TOPS EDTK XCUR ANY 
        IVDA NXL DTSS GWAV SDOT ENTO ISPO RVSN TPIC GOVX NLSP CMMB HOVR WAFU POAI VRME HKIT NUKK BFRI REBN AMS TENX CXAI 
        CXAI OPI INDO COSM AEMD SONM SNTG NCPL INDO TBH MWYN GXAI ATNF BGLC APWC DWTX HTCR TOPS CLDI BINI RENT SSY VRAR 
        IPWR SJ CLPS IXHL NXL ICON LPCN INDO SNTG CLPS PC ADTX FENG XCUR DYAI DHAI ANY JBDI HLP COSM SNSE DATS ITRM NLSP 
        ALBT GREE OPAD CLDI HKIT EONR BNAI ABVC RVSN BCTX INDO MRM LEDS IHT NERV BKYI MCVT MTC LPTX IMNN TRT WAFU QMCO 
        KITT BCDA CARV GTEC CARV SEED DATS UONEK CODX DTSS CMMB MFI HKPD MITQ QTTB CLIR POLA VVOS ATER TOPS GOVX MTEK 
        QMCO HUDI EFOI BFRG MNTS GDHG KOSS CISO ATER BCAB SIDU IGC GNPX NCNA BLBX MGIH CXAI SILO RENB EEIQ HYPD NNVC LTRN 
        QNRX ANTE COSM QNRX TPIC RENB QH GBR ADIL DFLI WLGS SFHG STAI TXMD BIYA BEEM HUDI AMST GIBO CODX CHSN BGI CLSD 
        AAME SOND IGC CXAI NCI MESA DRCT WFF BEEM DXYZ GREE AMBO PETZ WLGS ISPC NAKA EBON GPUS EPOW CXAI CXAI XTLB GNPX 
        BFRG WIMI MOGO PRTS SYTA EEIQ PW WIMI PPSI HYPD NXL CXAI DRCT DUO ARBK KITT PHIO SUNE KIDZ IPW PMCB GTEC LEDS 
        HUSA LGCL KALA SSY BMR TNFA ILAG GBR NUWE BGFV XIN ONMD APWC TPST ATXG BTOG GXAI GFAI JFU IMTE INBS VVPR PAVM 
        XHLD UUU FGF CYCN AIMD MXC CDIO CMMB TPIC DLPN IINN WORX AEMD CSAI MLSS ARQQW ENTO PRLD LITM BURU GREE ARBK IBO 
        ENTO ZKIN GSIW NINE BOF RVSN DTST CETX IMCC BTAI LMFA INTZ SILO WKHS ATNF NIVF INDO CXAI GP ENTO TOP SPCB QMCO 
        AIFF AWX LIXT STAI QH AZTR ZKIN ATER TPIC MLSS ALF MAIA CARV MTC GTI STAI AGMH CSAI PWM"""

        # Convert string to list
        tickers = tickers_str.split()

        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for ticker in tickers:
            if ticker not in seen:
                seen.add(ticker)
                unique_tickers.append(ticker)

        print(f"Total unique tickers: {len(unique_tickers)}")

        # You need to provide the gap dates for each ticker
        # For now, using placeholder dates - YOU MUST UPDATE THESE WITH ACTUAL GAP DATES
        gap_dates = ['2024-01-01'] * len(unique_tickers)  # REPLACE WITH ACTUAL DATES

        # Create DataFrame
        custom_df = pd.DataFrame({
            'ticker': unique_tickers,
            'gap_date': gap_dates
        })

        # Save to CSV
        custom_df.to_csv('custom_tickers.csv', index=False)
        print(f"Created custom_tickers.csv with {len(unique_tickers)} unique tickers")
        print("\n⚠️ IMPORTANT: You must update the gap_dates in custom_tickers.csv")
        print("Replace '2024-01-01' with the actual dates when each ticker gapped up 20%+")
        print("Then change INPUT_FILE = 'custom_tickers.csv' and run the analysis")
