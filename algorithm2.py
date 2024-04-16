import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict
import math
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."


logger = Logger()



class RecordedData: 
    def __init__(self):
        self.starfruit_cache = []
        '''
        self.POSITION_LIMIT = {
            'AMETHYSTS' : 20, 
            'STARFRUIT' : 20, 
            'ORCHIDS': 100,
        }
        self.INF = int(1e9)
        self.starfruit_dimension = 38
        self.AME_RANGE = 2
        self.position = {
            'AMETHYSTS' : 0, 
            'STARFRUIT' : 0, 
            'ORCHIDS': 0,
        }
        self.orchids_mm_spread = 5
        '''

class Trader:
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS': 100}
    INF = int(1e9)
    starfruit_dimension = 38
    AME_RANGE = 2
    orchids_mm_spread = 5
    position = {}
    round1_products = ['AMETHYSTS', 'STARFRUIT']
    round2_products = ['ORCHIDS']

    def estimate_starfruit_price(self, cache, alpha = 0.2):
        '''
        x = np.array([i for i in range(self.starfruit_dimension)])
        y = np.array(cache)
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return int(round(self.starfruit_dimension * m + c))
        '''
        res = cache[0]
        for price in cache[1:]:
            res = alpha * price + (1-alpha) * res
        return int(res)



    def values_extract(self, orders, buy):
        volume = 0
        best = 0 if buy else self.INF

        for price, vol in orders.items():
            if buy:
                volume += vol
                best = max(best, price)
            else:
                volume -= vol
                best = min(best, price)

        return volume, best
    

    # given estimated bid and ask prices, market take if there are good offers, otherwise market make 
    # by pennying or placing our bid/ask, whichever is more profitable
    def calculate_round1_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(sell_orders, buy=False)
        buy_vol, best_buy_price = self.values_extract(buy_orders, buy=True)

        logger.print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = self.position[product] 
        limit = self.POSITION_LIMIT[product]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = self.position[product]

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -limit-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 

        return orders


    def calculate_round2_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(sell_orders, buy=False)
        buy_vol, best_buy_price = self.values_extract(buy_orders, buy=True)

        logger.print(f'Product: {product} - best sell: {best_sell_price}, best buy: {best_buy_price}')

        position = 0
        limit = self.POSITION_LIMIT[product]

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(penny_sell, our_ask)
        ask_price = max(best_sell_price-self.orchids_mm_spread, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < limit and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, limit - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < limit:
            num_orders = limit - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -limit and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -limit-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -limit:
            num_orders = -limit - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 

        return orders

                      
    def calc_vwap(self, order_depth):
        buy_price = np.array([item for item in order_depth.buy_orders.keys() if not np.isnan(item)])
        buy_volume = np.array([item for item in order_depth.buy_orders.values() if not np.isnan(item)])
        sell_price = np.array([item for item in order_depth.sell_orders.keys() if not np.isnan(item)])
        sell_volume = -1 * np.array([item for item in order_depth.sell_orders.values() if not np.isnan(item)])

        vwap = float(((buy_price * buy_volume).sum() + (sell_price * sell_volume).sum()) / (buy_volume.sum() + sell_volume.sum()))

        return vwap

    
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        if state.traderData == "": # first run, set up data
            data = RecordedData()
        else:
            data = jsonpickle.decode(state.traderData)

        # self.POSITION_LIMIT = data.POSITION_LIMIT
        # self.INF = data.INF
        # self.starfruit_dimension = data.starfruit_dimension
        # self.AME_RANGE = data.AME_RANGE
        # self.position = data.position
        # self.orchids_mm_spread = data.orchids_mm_spread

        # update our position 
        for product in state.order_depths:
            self.position[product] = state.position[product] if product in state.position else 0


        # calculate bid/ask range
        if len(data.starfruit_cache) == self.starfruit_dimension:
            data.starfruit_cache.pop(0)
        _, best_sell_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())), False)
        _, best_buy_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse= True)), True)
        data.starfruit_cache.append((best_sell_starfruit + best_buy_starfruit) / 2)
        starfruit_lb, starfruit_ub = -self.INF, self.INF
        if len(data.starfruit_cache) == self.starfruit_dimension:
            starfruit_lb = self.estimate_starfruit_price(data.starfruit_cache)-2
            starfruit_ub = self.estimate_starfruit_price(data.starfruit_cache)+2
        acceptable_bids = {
            'AMETHYSTS': 10000 - self.AME_RANGE,
            'STARFRUIT': starfruit_lb,
            'ORCHIDS': 0,
        }

        acceptable_asks = {
            'AMETHYSTS': 10000+self.AME_RANGE,
            'STARFRUIT': starfruit_ub,
            'ORCHIDS': 0,
        }


        # round 1
        for product in self.round1_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            orders += self.calculate_round1_orders(product, order_depth, acceptable_bids[product], acceptable_asks[product])
            result[product] = orders
            logger.print(f'placed orders: {orders}')

        # round 2
        for product in self.round2_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            orchids_observation = state.observations.conversionObservations['ORCHIDS']

            real_bid = orchids_observation.bidPrice - orchids_observation.exportTariff - orchids_observation.transportFees - 0.1
            real_ask = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees

            lower_bound = int(round(real_ask))-1
            upper_bound = int(round(real_ask))+1

            orders += self.calculate_round2_orders(product, order_depth, lower_bound, upper_bound)
            conversions = -self.position[product]
            result[product] = orders
            logger.print(f'placed orders: {orders}')

        traderData = jsonpickle.encode(data)

        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData