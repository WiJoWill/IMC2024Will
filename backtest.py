import os
import pandas as pd
import statistics
from datetime import datetime
import copy
import uuid
from datamodel import *
from algorithm import *

import matplotlib.pyplot as plt
import seaborn as sns

'''
first_round_pst = ['PEARLS', 'BANANAS']
snd_round_pst = first_round_pst + ['COCONUTS',  'PINA_COLADAS']
third_round_pst = snd_round_pst + ['DIVING_GEAR', 'BERRIES']
fourth_round_pst = third_round_pst + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round_pst = fourth_round_pst # + secret, maybe pirate gold?

SYMBOLS_BY_ROUND_POSITIONABLE = {
    1: first_round_pst,
    2: snd_round_pst,
    3: third_round_pst,
    4: fourth_round_pst,
    5: fifth_round_pst,
}
'''

class BacktestingSystem:
    def __init__(self, training_data_prefix="./training", time_delta = 100):
        # General Configs
        self.TRAINING_DATA_PREFIX = training_data_prefix
        self.TIME_DELTA = time_delta

        # Asset Symbols/Tickers (positionable, some assets are used for indicators)
        # type: Dict (key = round number: int, value = [available symbols: str])
        self.ASSET_SYMBOLS_POSITIONABLE = {
            1: ['STARFRUIT', 'AMETHYSTS']
        }

        # Asset Symbols/Tickers
        # type: Dict (key = round number: int, value = [available symbols: str])
        self.ASSET_SYMBOLS = {
            1: ['STARFRUIT', 'AMETHYSTS']
        }

        # Limit for each asset
        # type: Dict (key = asset symbols: str, value = position limit: int)
        self.ASSET_LIMITS = {
            'STARFRUIT': 20,
            'AMETHYSTS': 20
        }

        # Other initialization parameters as needed

    def process_prices(self, df_prices, round, time_limit) -> dict[int, TradingState]:
        # Same as the previous process_prices function but now a method of the class
        # Remember to use self.TIME_DELTA where necessary
        
        states = {}
        for _, row in df_prices.iterrows():
            time: int = int(row["timestamp"])
            if time > time_limit:
                break
            product: str = row["product"]
            if states.get(time) == None:
                traderdata: str = ""
                position: Dict[Product, Position] = {}
                own_trades: Dict[Symbol, List[Trade]] = {}
                market_trades: Dict[Symbol, List[Trade]] = {}
                observations: Observation = Observation({}, {})
                listings = {}
                depths = {}
                states[time] = TradingState(traderdata, time, listings, depths, own_trades, market_trades, position, observations)

            if product not in states[time].position and product in self.ASSET_SYMBOLS_POSITIONABLE[round]:
                states[time].position[product] = 0
                states[time].own_trades[product] = []
                states[time].market_trades[product] = []

            states[time].listings[product] = Listing(product, product, "1")

            # Special Observation
            if product == "DOLPHIN_SIGHTINGS":
                states[time].observations["DOLPHIN_SIGHTINGS"] = row['mid_price']
                
            depth = OrderDepth()
            if row["bid_price_1"]> 0:
                depth.buy_orders[row["bid_price_1"]] = int(row["bid_volume_1"])
            if row["bid_price_2"]> 0:
                depth.buy_orders[row["bid_price_2"]] = int(row["bid_volume_2"])
            if row["bid_price_3"]> 0:
                depth.buy_orders[row["bid_price_3"]] = int(row["bid_volume_3"])
            if row["ask_price_1"]> 0:
                depth.sell_orders[row["ask_price_1"]] = -int(row["ask_volume_1"])
            if row["ask_price_2"]> 0:
                depth.sell_orders[row["ask_price_2"]] = -int(row["ask_volume_2"])
            if row["ask_price_3"]> 0:
                depth.sell_orders[row["ask_price_3"]] = -int(row["ask_volume_3"])
            states[time].order_depths[product] = depth

        return states


    def process_trades(self, df_trades, states: dict[int, TradingState], time_limit, names=True):
        # Same as the previous process_trades function but now a method of the class
        for _, trade in df_trades.iterrows():
            time: int = trade['timestamp']
            if time > time_limit:
                break
            symbol = trade['symbol']
            if symbol not in states[time].market_trades:
                states[time].market_trades[symbol] = []
            t = Trade(
                    symbol, 
                    trade['price'], 
                    trade['quantity'], 
                    str(trade['buyer']), 
                    str(trade['seller']),
                    time)
            states[time].market_trades[symbol].append(t)
        return states

    def calc_mid(self, states: dict[int, TradingState], round: int, time: int, max_time: int) -> dict[str, float]:
        # Same as the previous calc_mid function but now a method of the class
        # Remember to use self.TIME_DELTA where necessary
        medians_by_symbol = {}
        non_empty_time = time
        for psymbol in self.ASSET_SYMBOLS_POSITIONABLE[round]:
            hitted_zero = False
            while len(states[non_empty_time].order_depths[psymbol].sell_orders.keys()) == 0 or len(states[non_empty_time].order_depths[psymbol].buy_orders.keys()) == 0:
                # little hack
                if time == 0 or hitted_zero and time != max_time:
                    hitted_zero = True
                    non_empty_time += self.TIME_DELTA
                else:
                    non_empty_time -= self.TIME_DELTA
            min_ask = min(states[non_empty_time].order_depths[psymbol].sell_orders.keys())
            max_bid = max(states[non_empty_time].order_depths[psymbol].buy_orders.keys())
            median_price = statistics.median([min_ask, max_bid])
            medians_by_symbol[psymbol] = median_price
        return medians_by_symbol
    
    def cleanup_order_volumes(self, org_orders: List[Order]) -> List[Order]:
        orders = []
        for order_1 in org_orders:
            final_order = copy.copy(order_1)
            for order_2 in org_orders:
                if order_1.price == order_2.price and order_1.quantity == order_2.quantity:
                    continue 
                if order_1.price == order_2.price:
                    final_order.quantity += order_2.quantity
            orders.append(final_order)
        return orders
    
    def clear_order_book(self, trader_orders: dict[str, List[Order]], order_depth: dict[str, OrderDepth], time: int, halfway: bool) -> list[Trade]:
        trades = [] 
        # print(trader_orders) 
        for symbol in trader_orders.keys():
            if order_depth.get(symbol) != None:
                symbol_order_depth = copy.deepcopy(order_depth[symbol])
                t_orders = self.cleanup_order_volumes(trader_orders[symbol])
                for order in t_orders:
                    if order.quantity < 0:
                        if halfway:
                            bids = symbol_order_depth.buy_orders.keys()
                            asks = symbol_order_depth.sell_orders.keys()
                            max_bid = max(bids)
                            min_ask = min(asks)
                            if order.price <= statistics.median([max_bid, min_ask]):
                                trades.append(Trade(symbol, order.price, order.quantity, "BOT", "YOU", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                        else:
                            potential_matches = list(filter(lambda o: o[0] == order.price, symbol_order_depth.buy_orders.items()))
                            if len(potential_matches) > 0:
                                match = potential_matches[0]
                                final_volume = 0
                                if abs(match[1]) > abs(order.quantity):
                                    final_volume = order.quantity
                                else:
                                    #this should be negative
                                    final_volume = -match[1]
                                trades.append(Trade(symbol, order.price, final_volume, "BOT", "YOU", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                    if order.quantity > 0:
                        if halfway:
                            bids = symbol_order_depth.buy_orders.keys()
                            asks = symbol_order_depth.sell_orders.keys()
                            max_bid = max(bids)
                            min_ask = min(asks)
                            if order.price >= statistics.median([max_bid, min_ask]):
                                trades.append(Trade(symbol, order.price, order.quantity, "YOU", "BOT", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
                        else:
                            potential_matches = list(filter(lambda o: o[0] == order.price, symbol_order_depth.sell_orders.items()))
                            if len(potential_matches) > 0:
                                match = potential_matches[0]
                                final_volume = 0
                                #Match[1] will be negative so needs to be changed to work here
                                if abs(match[1]) > abs(order.quantity):
                                    final_volume = order.quantity
                                else:
                                    final_volume = abs(match[1])
                                trades.append(Trade(symbol, order.price, final_volume, "YOU", "BOT", time))
                            else:
                                print(f'No matches for order {order} at time {time}')
                                print(f'Order depth is {order_depth[order.symbol].__dict__}')
        return trades
                            
    
    def trades_position_pnl_run(self, states: dict[int, TradingState], max_time: int, profits_by_symbol: dict[int, dict[str, float]], 
        balance_by_symbol: dict[int, dict[str, float]], credit_by_symbol: dict[int, dict[str, float]], unrealized_by_symbol: dict[int, dict[str, float]], halfway: str):
        for time, state in states.items():
            position = copy.deepcopy(state.position)
            orders, conversion, traderData  = trader.run(state)
            trades = self.clear_order_book(orders, state.order_depths, time, halfway)
            mids = self.calc_mid(states, round, time, max_time)
            if profits_by_symbol.get(time + self.TIME_DELTA) == None and time != max_time:
                profits_by_symbol[time + self.TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + self.TIME_DELTA) == None and time != max_time:
                credit_by_symbol[time + self.TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + self.TIME_DELTA) == None and time != max_time:
                balance_by_symbol[time + self.TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + self.TIME_DELTA) == None and time != max_time:
                unrealized_by_symbol[time + self.TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for psymbol in self.ASSET_SYMBOLS_POSITIONABLE[round]:
                    unrealized_by_symbol[time + self.TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
            valid_trades = []
            failed_symbol = []
            grouped_by_symbol = {}
            if len(trades) > 0:
                for trade in trades:
                    if trade.symbol in failed_symbol:
                        continue
                    n_position = position[trade.symbol] + trade.quantity
                    if abs(n_position) > self.ASSET_LIMITS[trade.symbol]:
                        print('ILLEGAL TRADE, WOULD EXCEED POSITION LIMIT, KILLING ALL REMAINING ORDERS')
                        trade_vars = vars(trade)
                        trade_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                        print(f'Stopped at the following trade: {trade_str}')
                        print(f"All trades that were sent:")
                        for trade in trades:
                            trade_vars = vars(trade)
                            trades_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                            print(trades_str)
                        failed_symbol.append(trade.symbol)
                    else:
                        valid_trades.append(trade) 
                        position[trade.symbol] += trade.quantity
            FLEX_TIME_DELTA = self.TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                    if grouped_by_symbol.get(valid_trade.symbol) == None:
                        grouped_by_symbol[valid_trade.symbol] = []
                    grouped_by_symbol[valid_trade.symbol].append(valid_trade)
                    credit_by_symbol[time + FLEX_TIME_DELTA][valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].own_trades = grouped_by_symbol
                for psymbol in self.ASSET_SYMBOLS_POSITIONABLE[round]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol] = mids[psymbol]*position[psymbol]
                    if position[psymbol] == 0 and states[time].position[psymbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][psymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] #+unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][psymbol] = credit_by_symbol[time + FLEX_TIME_DELTA][psymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][psymbol]

            if time == max_time:
                print("End of simulation reached. All positions left are liquidated")
                # i have the feeling this already has been done, and only repeats the same values as before
                for osymbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][osymbol] += credit_by_symbol[time + FLEX_TIME_DELTA][osymbol] + unrealized_by_symbol[time + FLEX_TIME_DELTA][osymbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][osymbol] = 0
            if states.get(time + FLEX_TIME_DELTA) != None:
                states[time + FLEX_TIME_DELTA].position = copy.deepcopy(position)
        return states, trader, profits_by_symbol, balance_by_symbol


    def simulate_alternative(self, round: int, day: int, trader, time_limit=999900, names=False, halfway = False, other_traders = False, other_trader_names=['Caesar', 'Camilla', 'Peter']):
        # Same as the previous simulate_alternative function but now a method of the class
        # Adjustments: Replace global constants and functions with self.attribute or self.method()
        
        prices_path = os.path.join(self.TRAINING_DATA_PREFIX, f'prices_round_{round}_day_{day}.csv')
        trades_path = os.path.join(self.TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_wn.csv')
        if not names:
            trades_path = os.path.join(self.TRAINING_DATA_PREFIX, f'trades_round_{round}_day_{day}_nn.csv')
        
        df_prices = pd.read_csv(prices_path, sep=';')
        df_trades = pd.read_csv(trades_path, sep=';', dtype = {'seller': str, 'buyer': str })

        states = self.process_prices(df_prices, round, time_limit)
        states = self.process_trades(df_trades, states, time_limit, names)
        ref_symbols = list(states[0].position.keys())
        max_time = max(list(states.keys()))

        # handling these four is rather tricky 
        profits_by_symbol: dict[int, dict[str, float]] = { 0: dict(zip(ref_symbols, [0.0]*len(ref_symbols))) }
        balance_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
        credit_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }
        unrealized_by_symbol: dict[int, dict[str, float]] = { 0: copy.deepcopy(profits_by_symbol[0]) }

        states, trader, profits_by_symbol, balance_by_symbol = self.trades_position_pnl_run(states, max_time, profits_by_symbol, balance_by_symbol, credit_by_symbol, unrealized_by_symbol, halfway)
        self.create_log_files(round, day, states, profits_by_symbol, balance_by_symbol, trader)
        # profit_balance_monkeys = {}
        # trades_monkeys = {}
        '''
        if other_traders:
            
            profit_balance_monkeys, trades_monkeys, profit_monkeys, balance_monkeys, monkey_positions_by_timestamp = monkey_positions(other_trader_names, states, round)
            print("End of monkey simulation reached.")
            print(f'PNL + BALANCE monkeys {profit_balance_monkeys[max_time]}')
            print(f'Trades monkeys {trades_monkeys[max_time]}')
        '''
        if hasattr(trader, 'after_last_round'):
            if callable(trader.after_last_round): #type: ignore
                trader.after_last_round(profits_by_symbol, balance_by_symbol) #type: ignore
    # Additional methods for cleanup_order_volumes, clear_order_book, create_log_file, etc., similarly converted
        
        self.build_plots(profits_by_symbol, f'profits on round{round}-day{day}')

    def create_log_files(self, round: int, day: int, states: dict[int, TradingState], profits_by_symbol: dict[int, dict[str, float]], balance_by_symbol: dict[int, dict[str, float]], trader: Trader):
        file_name = uuid.uuid4()
        timest = datetime.timestamp(datetime.now())
        max_time = max(list(states.keys()))
        log_path = os.path.join('logs', f'{timest}_{file_name}.log')

        csv_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
        with open(log_path, 'w', encoding="utf-8", newline='\n') as f:
            f.write('\n')
            for time, state in states.items():
                if hasattr(trader, 'logger'):
                    if hasattr(trader.logger, 'local_logs') != None:
                        if trader.logger.local_logs.get(time) != None:
                            f.write(f'{time} {trader.logger.local_logs[time]}\n')
                            continue
                if time != 0:
                    f.write(f'{time}\n')

            f.write(f'\n\n')
            f.write('Submission logs:\n\n\n')
            f.write('Activities log:\n')
            f.write(csv_header)
            for time, state in states.items():
                for symbol in self.ASSET_SYMBOLS[round]:
                    f.write(f'{day};{time};{symbol};')
                    bids_length = len(state.order_depths[symbol].buy_orders)
                    bids = list(state.order_depths[symbol].buy_orders.items())
                    bids_prices = list(state.order_depths[symbol].buy_orders.keys())
                    bids_prices.sort()
                    asks_length = len(state.order_depths[symbol].sell_orders)
                    asks_prices = list(state.order_depths[symbol].sell_orders.keys())
                    asks_prices.sort()
                    asks = list(state.order_depths[symbol].sell_orders.items())
                    if bids_length >= 3:
                        f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};{bids[2][0]};{bids[2][1]};')
                    elif bids_length == 2:
                        f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};;;')
                    elif bids_length == 1:
                        f.write(f'{bids[0][0]};{bids[0][1]};;;;;')
                    else:
                        f.write(f';;;;;;')
                    if asks_length >= 3:
                        f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};{asks[2][0]};{asks[2][1]};')
                    elif asks_length == 2:
                        f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};;;')
                    elif asks_length == 1:
                        f.write(f'{asks[0][0]};{asks[0][1]};;;;;')
                    else:
                        f.write(f';;;;;;')


                    if len(asks_prices) == 0 or max(bids_prices) == 0:
                        if symbol == 'DOLPHIN_SIGHTINGS':
                            dolphin_sightings = state.observations['DOLPHIN_SIGHTINGS']
                            f.write(f'{dolphin_sightings};{0.0}\n')
                        else:
                            f.write(f'{0};{0.0}\n')
                    else:
                        actual_profit = 0.0
                        if symbol in self.ASSET_SYMBOLS_POSITIONABLE[round]:
                                actual_profit = profits_by_symbol[time][symbol] + balance_by_symbol[time][symbol]
                        min_ask = min(asks_prices)
                        max_bid = max(bids_prices)
                        median_price = statistics.median([min_ask, max_bid])
                        f.write(f'{median_price};{actual_profit}\n')
                        if time == max_time:
                            if profits_by_symbol[time].get(symbol) != None:
                                print(f'Final profit for {symbol} = {actual_profit}')
        print(f"\nSimulation on round {round} day {day} for time {max_time} complete")
    

    def build_plots(self, data: dict[int, dict[str, float]], filename: str = 'No filename', path = 'plots/'):
        products = list(self.ASSET_LIMITS.keys())

        fig, axs = plt.subplots(len(products) + 1, 1, figsize=(10, 5 * len(products)), sharex=True)

        if len(products) == 1:
            axs = [axs]
        
        timestamps = []
        total_values = {}
        for idx, product in enumerate(products):
            timestamps = []
            values = []
            for timestamp, timestamp_data in data.items():
                if product in timestamp_data:
                    timestamps.append(timestamp)
                    values.append(timestamp_data[product])
            total_values[product] = copy.copy(values)
            axs[idx].plot(timestamps, values, label=f'{product} Value')
            axs[idx].set_title(f'Values of {product}')
            axs[idx].set_ylabel('Value')
            axs[idx].legend()

            axs[idx].annotate(
                f'{values[-1]}',
                xy = (timestamps[-1], values[-1]),
                fontsize = 3,
                ha='center'
            )

        for product in total_values:
            axs[-1].plot(timestamps, total_values[product], label = f'{product} Value')
            axs[-1].set_title('Values of  All Products')
            axs[-1].set_ylabel('Value')
            axs[-1].legend()

            
            axs[-1].annotate(
                f'{product}: {total_values[product][-1]}',
                xy = (timestamps[-1], total_values[product][-1]),
                ha='center'
            )


        axs[-1].set_xlabel('Timestamp')
        plt.tight_layout()
        plt.savefig(path + filename, bbox_inches = 'tight')
        plt.close(fig)


if __name__ == "__main__":
    # Example usage
    trader = Trader()  # Assuming Trader is defined elsewhere as per the provided snippets
    backtest_system = BacktestingSystem()
    #os.chdir('IMC2024Will')
    round = 1  # Example parameters
    day = 0
    backtest_system.simulate_alternative(round, day, trader)
