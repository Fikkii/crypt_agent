import os
import time
from crewai.tools import tool
from pybit.unified_trading import HTTP
import pandas as pd

from typing import List

# Securely initialize session
session = HTTP(
    testnet=False,
    demo=True,
    api_key=os.getenv("BYBIT_DEMO_API_KEY"),
    api_secret=os.getenv("BYBIT_DEMO_API_SECRET"),
    recv_window=10000
)

@tool("get_latest_klines")
def get_latest_klines(symbol="BTCUSDT", interval="1"):
    """ Fetches the latest kline for the specified coin
        
        Required args:
            symbol (string): symbol Name.
            interval (string): kline interval. 1,3,5,15,30,60,120,240,360,720,D,M,W

        Returns results as dictionary
    """
    # Use pybit to get kline data
    return session.get_kline(category="linear", symbol=symbol, limit=100, interval=interval)

@tool("calculate_technical_indicators")
def calculate_technical_indicators(symbol: str):
    """
        Fetches historical kline data for a symbol and calculates 
        RSI and MACD indicators using pandas.

        Required args:
            symbol (string): symbol Name.
            interval (string): kline interval. 1,3,5,15,30,60,120,240,360,720,D,M,W
    """
    try:
        # 1. Fetch 100 candles (enough for RSI-14 and MACD-26)
        kline_data = session.get_kline(
            category="linear",
            symbol=symbol,
            interval="15", # 15-minute timeframe
            limit=100
        )
        
        # 2. Extract the list of candles from the Bybit response
        # Bybit returns: [startTime, open, high, low, close, volume, turnover]
        raw_list = kline_data['result']['list']
        
        # 3. Create a Pandas DataFrame
        df = pd.DataFrame(raw_list, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
        ])
        
        # Convert 'close' to numeric (Bybit returns strings)
        df['close'] = pd.to_numeric(df['close'])
        
        # Bybit returns data from NEWEST to OLDEST, we need to flip it for pandas
        df = df.iloc[::-1].reset_index(drop=True)

        # 4. CALCULATE RSI (14-period)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # 5. CALCULATE MACD (12, 26, 9)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['macd'] - df['signal']

        # 6. Get the most recent values (the last row)
        latest = df.iloc[-1]
        
        return {
            "symbol": symbol,
            "current_price": latest['close'],
            "rsi": round(latest['rsi'], 2),
            "macd_value": round(latest['macd'], 4),
            "macd_signal": round(latest['signal'], 4),
            "macd_histogram": round(latest['histogram'], 4),
            "status": "success"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool("math_calculator")
def math_tool(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result."""
    try:
        result = eval(expression)
        return f"The result of '{expression}' is: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool("request_demo_funds")
def request_demo_funds() -> str:
    """Requests demo trading funds for the Unified Trading Account."""
    try:
        response = session.request_demo_trading_funds()
        return f"Demo Trading Funds Response: {response['result']}"
    except Exception as e:
        return f"Error requesting demo funds: {str(e)}"

@tool("check_wallet_balance")
def check_wallet_balance(coin: str = "USDT") -> str:
    """Checks the balance of a specific coin in the Unified Trading Account."""
    try:
        response = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        data = response['result']['list'][0]['coin'][0]
        return f"Current {coin} Balance: {data['walletBalance']}"
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


@tool("execute_multiple_orders")
def execute_multiple_orders(orders: List[dict]):
    """
    Receives an array of order objects and executes them.

    Required arg:
        orders(list): array of order objects, where each object contains:
            - symbol (string): e.g., "BTCUSDT"
            - side (string): "Buy" or "Sell"
            - quantity (string): Order quantity
    """
    results = []
    for order in orders:
        try:
            response = session.place_order(
                category="linear",
                symbol=order['symbol'],
                side=order['side'],
                orderType="Market",
                qty=order['quantity'],
            )
            results.append(f"Success: {order['symbol']} OrderID: {response['result']['orderId']}")
        except Exception as e:
            results.append(f"Failed: {order['symbol']} Error: {str(e)}")
    
    return "\n".join(results)

@tool("place_market_order")
def place_market_order(symbol: str, side: str, qty: str,  tp_price: str, sl_price: str, category: str = "linear" ) -> str:
    """This method supports to place order for spot and linear.
        
        Required args:
            symbol(string): symbol name
            side(string): Buy, Sell
            qty(string): Order quantity
            tp_price(string): Take profit price (e.g., "70000.50")
            sl_price(string): Stop loss price (e.g., "65000.00")
    """
    try:
        # 1. SYMBOL VALIDATION (The Failsafe)
        instruments = session.get_instruments_info(
            category="linear",
            symbol=symbol
        )
        
        # Check if the result list is empty (symbol doesn't exist)
        if not instruments['result']['list']:
            return f"Error: '{symbol}' is not a valid Bybit symbol for the linear category."
        
        instrument_info = instruments['result']['list'][0]
        
        # Check if the coin is currently tradable
        if instrument_info['status'] != "Trading":
            return f"Error: {symbol} exists but is currently {instrument_info['status']}."
        response = session.place_order(
            category=category,
            symbol=symbol,
            side=side,
            orderType="Market",
            qty=qty,
            takeProfit=tp_price, # e.g., "70000.50"
            stopLoss=sl_price,   # e.g., "65000.00"
            tpTriggerBy="MarkPrice", # Recommended for safety
            slTriggerBy="MarkPrice",
            tpslMode="Full", # Entire position closes when hit
        )
        return f"Success! {side} {qty} {symbol}. Order ID: {response['result']['orderId']}"
    except Exception as e:
        return f"Trade failed: {str(e)}"

@tool("fetch_ticker_price")
def fetch_ticker_price(symbol: str, category: str = "linear") -> str:
    """Fetches real-time price of a ticker.

        Required args:
            category(string): spot, linear
            symbol(string): symbol name
    """
    try:
        response = session.get_tickers(category=category, symbol=symbol)
        price = response['result']['list'][0]['lastPrice']
        return f"Current price of {symbol}: {price}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":


    print(f'ENV keys: {os.getenv("BYBIT_DEMO_API_KEY")}, {os.getenv("BYBIT_DEMO_API_SECRET")}')
    coin = "USDT"
    response = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
    data = response['result']['list'][0]['coin'][0]
    print(f"Current {coin} Balance: {data['walletBalance']}")

    # import time
    #
    # session = HTTP(testnet=True)
    # server_time = int(session.get_server_time()['result']['timeNano']) / 1_000_000
    # local_time = time.time() * 1000
    #
    # print(f"Server Time: {server_time}")
    # print(f"Local Time:  {local_time}")
    # print(f"Time Diff:   {local_time - server_time} ms")



@tool("advanced_sliced_executor")
def advanced_sliced_executor(symbol: str, side: str, total_qty: float, tp_price: str, sl_price: str):
    """
    Executes trades on Bybit Perpetuals. If the total_qty exceeds the 
    exchange's maximum allowed per order, it automatically slices the 
    trade into multiple valid orders.

        Required args:
            symbol(string): symbol name
            side(string): Buy, Sell
            total_qty(string): Order quantity
            tp_price(string): Take profit price (e.g., "70000.50")
            sl_price(string): Stop loss price (e.g., "65000.00")
    """
    try:
        # 1. Fetch the max limits for the specific coin
        instrument = session.get_instruments_info(
            category="linear",
            symbol=symbol
        )
        
        if not instrument['result']['list']:
            return f"Error: {symbol} not found."

        lot_filter = instrument['result']['list'][0]['lotSizeFilter']
        max_order_qty = float(lot_filter['maxOrderQty'])
        min_order_qty = float(lot_filter['minOrderQty'])
        
        # 2. Safety Check: Is the order too small?
        if total_qty < min_order_qty:
            return f"Failed: Quantity {total_qty} is below the minimum required ({min_order_qty})."

        # 3. Execution Logic: Slicing the Order
        orders_executed = []
        remaining_qty = total_qty
        
        print(f"Executing total {side} order for {total_qty} {symbol}. Max per order: {max_order_qty}")

        while remaining_qty > 0:
            # Determine size for this specific slice
            current_slice_qty = min(remaining_qty, max_order_qty)
            
            # Place the order slice
            order = session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=current_slice_qty,
                takeProfit=tp_price, # e.g., "70000.50"
                stopLoss=sl_price,   # e.g., "65000.00"
                tpTriggerBy="MarkPrice", # Recommended for safety
                slTriggerBy="MarkPrice",
                tpslMode="Full", # Entire position closes when hit
                timeInForce="GTC"
            )
            
            order_id = order['result']['orderId']
            orders_executed.append(f"{current_slice_qty} units (ID: {order_id})")
            
            # Update remaining amount
            remaining_qty -= current_slice_qty
            
            # Small sleep to prevent hitting rate limits during rapid slicing
            if remaining_qty > 0:
                time.sleep(0.1) 

        return f"SUCCESS: Fully executed {total_qty} {symbol} in {len(orders_executed)} slices:\n" + "\n".join(orders_executed)

    except Exception as e:
        return f"CRITICAL FAILURE: {str(e)}"

