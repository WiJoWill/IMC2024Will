import json
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, OrderedDict
import collections
import numpy as np
import pandas as pd
import math

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
            # self.truncate(trader_data, max_item_length),
            # self.truncate(self.logs, max_item_length),
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
            try:
                compressed.append([listing["symbol"], listing["product"], listing["denomination"]])
            except:
                compressed.append([listing.symbol, listing.product, listing.denomination])

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
        # self.basket_spread_cache = []
        self.strawberries_cache = []
        self.coconut_coupon_implied_vol_cache = []
        self.coconut_coupon_implied_vol_size = 1000

class Trader:
    POSITION_LIMIT = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
        'GIFT_BASKET': 60,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
        'COCONUT_COUPON': 600, 
        'COCONUT': 300,
    }
    position = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0,
        'ORCHIDS': 0,
        'GIFT_BASKET': 0,
        'CHOCOLATE': 0,
        'STRAWBERRIES': 0,
        'ROSES': 0,
        'COCONUT_COUPON': 0, 
        'COCONUT': 0,
    }
    round1_products = ['AMETHYSTS', 'STARFRUIT']
    round2_products = ['ORCHIDS']
    round3_products = ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
    round4_products = ['COCONUT_COUPON', 'COCONUT']

    INF = int(1e9)

    starfruit_dimension = 35
    AME_RANGE = 0

    orchids_mm_spread = 5
    
    basket_premium = 370.27 # 379.5
    basket_premium_std = 79.4 # 76.43
    basket_premium_mean = 9.5
    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    basket_maxedge = 250
    strawberry_momentum_signal = 0
    strawberry_long_window = 1500

    coconut_coupon_implied_vol = 0.010043927128326914
    coconut_coupon_z_score = 1.2

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
    

    def black_scholes_price(self, S, K = 10000, t = 250,  r = 0, sigma = coconut_coupon_implied_vol, delta = False):
        def cdf(x):
            return 0.5 * (1 + math.erf(x/math.sqrt(2)))

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        price = S * cdf(d1) - K * np.exp(-r * t) * cdf(d2)

        if delta:
            return price, cdf(d1)

        return price
    

    def newtons_method(self, f, x0=0.02, epsilon=1e-7, max_iter=100, h=1e-5):
        def numerical_derivative(f, x, h=1e-5):
            return (f(x + h) - f(x - h)) / (2 * h)
        
        x = x0
        for i in range(max_iter):
            fx = f(x)
            if abs(fx) < epsilon:
                return x
            dfx = numerical_derivative(f, x, h)
            if dfx == 0:
                raise ValueError("Derivative zero. No solution found.")
            x -= fx / dfx
        raise ValueError("Maximum iterations reached. No solution found.")

    def estimate_starfruit_price(self, cache, alpha = 0.2):
        data = pd.Series(cache)
        predicted_price = data.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
        return int(round(predicted_price))

    # given estimated bid and ask prices, market take if there are good offers, otherwise market make 
    # by pennying or placing our bid/ask, whichever is more profitable
    def calculate_round1_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(sell_orders, buy=False)
        buy_vol, best_buy_price = self.values_extract(buy_orders, buy=True)

        position = self.position[product] 

        # penny the current highest bid / lowest ask 
        undercut_buy = best_buy_price + 1
        undercut_sell = best_sell_price - 1

        if product == 'AMETHYSTS':
            bid_price = min(undercut_buy, our_bid - 1)
            sell_price = max(undercut_sell, our_ask + 1)

            # Search the orders that satisfy our acceptable price
            # ask
            for ask, vol in sell_orders.items():
                if ((ask < our_bid) or ((self.position[product] < 0) and (ask == our_bid))) and position < self.POSITION_LIMIT[product]:
                    order_for = min(-vol, self.POSITION_LIMIT[product])
                    position += order_for
                    orders.append(Order(product, ask, order_for))
            
            if (position < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
                num = min(40, self.POSITION_LIMIT[product] - position)
                orders.append(Order(product, min(undercut_buy + 1, our_bid - 1), num))
                position += num

            if (position < self.POSITION_LIMIT[product]) and (self.position[product] > 15):
                num = min(40, self.POSITION_LIMIT[product] - position)
                orders.append(Order(product, min(undercut_buy - 1, our_bid - 1), num))
                position += num

            if position < self.POSITION_LIMIT[product]:
                num = min(40, self.POSITION_LIMIT[product] - position)
                orders.append(Order(product, bid_price, num))
                position += num

            position = self.position[product]

            for bid, vol in buy_orders.items():
                if ((bid > our_ask) or ((self.position[product] > 0) and (bid == our_ask))) and position > -self.POSITION_LIMIT[product]:
                    order_for = max(-vol, -self.POSITION_LIMIT[product]-position)
                    # order_for is a negative number denoting how much we will sell
                    position += order_for
                    assert(order_for <= 0)
                    orders.append(Order(product, bid, order_for))

            if (position > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
                num = max(-40, -self.POSITION_LIMIT[product]-position)
                orders.append(Order(product, max(undercut_sell - 1, our_ask + 1), num))
                position += num

            if (position > -self.POSITION_LIMIT[product]) and (self.position[product] < -15):
                num = max(-40, -self.POSITION_LIMIT[product]-position)
                orders.append(Order(product, max(undercut_sell + 1, our_ask + 1), num))
                position += num

            if position > -self.POSITION_LIMIT[product]:
                num = max(-40, -self.POSITION_LIMIT[product] - position)
                orders.append(Order(product, sell_price, num))
                position += num
        
        elif product == 'STARFRUIT':
            bid_price = min(undercut_buy, our_bid)
            ask_price = max(undercut_sell, our_ask)

            # MARKET TAKE ASKS (buy items)
            for ask, vol in sell_orders.items():
                if position < self.POSITION_LIMIT[product] and (ask <= our_bid or (position < 0 and ask == our_bid + 1)): 
                    num_orders = min(-vol, self.POSITION_LIMIT[product] - position)
                    position += num_orders
                    orders.append(Order(product, ask, num_orders))

            # MARKET MAKE 
            if position < self.POSITION_LIMIT[product]:
                num_orders = self.POSITION_LIMIT[product] - position
                orders.append(Order(product, bid_price, num_orders))
                position += num_orders

            position = self.position[product]

            # MARKET TAKE BIDS (sell items)
            for bid, vol in buy_orders.items():
                if position > -self.POSITION_LIMIT[product] and (bid >= our_ask or (position > 0 and bid == our_ask - 1)):
                    num_orders = max(-vol, -self.POSITION_LIMIT[product] - position)
                    position += num_orders
                    orders.append(Order(product, bid, num_orders))

            # MARKET MAKE
            if position > -self.POSITION_LIMIT[product]:
                num_orders = -self.POSITION_LIMIT[product] - position
                orders.append(Order(product, ask_price, num_orders))
                position += num_orders 
        
        return orders


    def calculate_round2_orders(self, product, order_depth, our_bid, our_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(sell_orders, buy=False)
        buy_vol, best_buy_price = self.values_extract(buy_orders, buy=True)

        position = 0

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(best_sell_price - self.orchids_mm_spread, our_ask)

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < self.POSITION_LIMIT[product] and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, self.POSITION_LIMIT[product] - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < self.POSITION_LIMIT[product]:
            num_orders = self.POSITION_LIMIT[product] - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -self.POSITION_LIMIT[product] and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -self.POSITION_LIMIT[product]-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -self.POSITION_LIMIT[product]:
            num_orders = -self.POSITION_LIMIT[product] - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 

        return orders
    
    def calculate_round2_orders_v2(self, product, order_depth, real_bid, real_ask):
        orders: list[Order] = []
        
        sell_orders = OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = self.values_extract(sell_orders, buy=False)
        buy_vol, best_buy_price = self.values_extract(buy_orders, buy=True)

        # penny the current highest bid / lowest ask 
        penny_buy = best_buy_price+1
        penny_sell = best_sell_price-1

        our_bid, our_ask = real_ask - 1, real_ask + 1

        bid_price = min(penny_buy, our_bid)
        ask_price = max(best_sell_price - self.orchids_mm_spread, our_ask)

        # case 1: local ask < south bid (buy local, sell south)
        # case 2: south bid < local bid < south ask < local ask - spread (one side mm) (buy south, ask local)
        # case 3: south bid < south ask < local bid < local ask (sell local, buy south)
        # case 4: local bid + spread < south bid (sell south, bid local)

        # case 1
        position = 0
        for ask, vol in sell_orders.items():
            if position < self.POSITION_LIMIT[product] and (ask <= real_bid):
                num_orders = min(-vol, self.POSITION_LIMIT[product] - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        position = 0
        # case 3
        for bid, vol in buy_orders.items():
            if position > self.POSITION_LIMIT[product] and (bid >= real_ask):
                num_orders = max(-vol, -self.POSITION_LIMIT[product]-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))
        
        # case 2
        '''
        for ask, vol in sell_orders.items():
            if (best_buy_price < ask - self.orchids_mm_spread) and (real_ask < ask - self.orchids_mm_spread < best_sell_price):
                orders.append(Order(product, ask - self.orchids_mm_spread, vol))
        '''
        if (best_buy_price < best_sell_price - self.orchids_mm_spread) and (real_ask < best_sell_price - self.orchids_mm_spread):
            orders.append(Order(product, best_sell_price - self.orchids_mm_spread, -buy_vol))

        
        # case 4
        '''
        for bid, vol in buy_orders.items():
            if (bid + self.orchids_mm_spread < best_sell_price) and (best_buy_price < bid + self.orchids_mm_spread < real_bid):
                orders.append(Order(product, bid + self.orchids_mm_spread, vol))
        '''
    
        if (best_buy_price + self.orchids_mm_spread < best_sell_price) and (best_buy_price + self.orchids_mm_spread < real_bid):
            orders.append(Order(product, best_buy_price + self.orchids_mm_spread, -sell_vol))
        
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

        # update our position 
        for product in state.order_depths:
            self.position[product] = state.position[product] if product in state.position else 0
        
        # round 1
        # calculate bid/ask range
        if len(data.starfruit_cache) == self.starfruit_dimension:
            data.starfruit_cache.pop(0)
        _, best_sell_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())), False)
        _, best_buy_starfruit = self.values_extract(OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse= True)), True)
        data.starfruit_cache.append((best_sell_starfruit + best_buy_starfruit) / 2)
        starfruit_lb, starfruit_ub = -self.INF, self.INF
        if len(data.starfruit_cache) == self.starfruit_dimension:
            starfruit_lb = self.estimate_starfruit_price(data.starfruit_cache) - 1
            starfruit_ub = self.estimate_starfruit_price(data.starfruit_cache) + 1
        acceptable_bids = {
            'AMETHYSTS': 10000 - self.AME_RANGE,
            'STARFRUIT': starfruit_lb,
            'ORCHIDS': 0,
        }

        acceptable_asks = {
            'AMETHYSTS': 10000 + self.AME_RANGE,
            'STARFRUIT': starfruit_ub,
            'ORCHIDS': 0,
        }
        
        for product in self.round1_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            orders += self.calculate_round1_orders(product, order_depth, acceptable_bids[product], acceptable_asks[product])
            result[product] = orders

        # round 2
        
        for product in self.round2_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders: list[Order] = []

            orchids_observation = state.observations.conversionObservations['ORCHIDS']

            real_bid = orchids_observation.bidPrice - orchids_observation.exportTariff - orchids_observation.transportFees - 0.1
            real_ask = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees

            lower_bound = int(round(real_ask)) - 1
            upper_bound = int(round(real_ask)) + 1

            # orders += self.calculate_round2_orders(product, order_depth, lower_bound, upper_bound)
            orders += self.calculate_round2_orders_v2(product, order_depth, real_bid, real_ask)
            conversions = -self.position[product]
            result[product] = orders
        
        #round 3
        if len(data.strawberries_cache) == self.strawberry_long_window: data.strawberries_cache.pop(0)
        orders = self.compute_orders_basket(state.order_depths, data)
        for product in self.round3_products:
            result[product] = orders[product]

        
        # round 4
        orders = self.calculate_round4_orders_v2(state.order_depths, data)
        for product in self.round4_products:
            result[product] = orders[product]


        traderData = jsonpickle.encode(data)

        logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
    
    def calculate_round4_orders(self, order_depths: dict[str, OrderDepth], data = None)-> dict[str, list]:
        prods = self.round4_products
        orders = {p: [] for p in prods}
        osell, obuy, best_sell, best_buy, mid_price = {}, {}, {}, {}, {}


        for p in prods:
            order_depth = order_depths[p]

            osell[p] = collections.OrderedDict(sorted(order_depths[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depths[p].buy_orders.items(), reverse=True))
            
            sell_vol, best_sell_price = self.values_extract(osell[p], buy=False)
            buy_vol, best_buy_price = self.values_extract(obuy[p], buy=True)

            best_sell[p] = best_sell_price
            best_buy[p] = best_buy_price
            mid_price[p] = (best_buy_price + best_sell_price) / 2

        coconut_mid, coupon_mid = mid_price['COCONUT'], mid_price['COCONUT_COUPON']
        implied_vol = self.newtons_method(lambda sigma: self.black_scholes_price(coconut_mid, 10000, 250, 0, sigma) - coupon_mid)
        # data.coconut_coupon_implied_vol_cache.append(implied_vol)

        # if len(data.coconut_coupon_implied_vol_cache) == data.coconut_coupon_implied_vol_size + 1:
            # data.coconut_coupon_implied_vol_cache.pop(0)
        implied_vol_mean, implied_vol_std = 0.010043927128326914, 0.00022995153208052514# np.mean(data.coconut_coupon_implied_vol_cache), np.std(data.coconut_coupon_implied_vol_cache)
        fair_coupon_price, delta = self.black_scholes_price(S = coconut_mid, sigma = implied_vol_mean, delta = True)

        z_score = (implied_vol - implied_vol_mean) / implied_vol_std


        if z_score > self.coconut_coupon_z_score: # over priced, let's sell coupon buy coconut
            coupon_position = self.position['COCONUT_COUPON']
            fair_ask = best_buy['COCONUT_COUPON']
            coupon_num_orders = 0
            for bid, vol in obuy['COCONUT_COUPON'].items():
                if coupon_position > -self.POSITION_LIMIT['COCONUT_COUPON'] and (bid >= fair_ask or (coupon_position > 0 and bid + 1 == fair_ask)):
                    num_orders = max(-vol, - self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                    coupon_position += num_orders
                    coupon_num_orders += num_orders
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', bid, num_orders))
            
            coconut_position, delta_hedging_vol = self.position['COCONUT'], -coupon_num_orders * delta
            for ask, vol in osell['COCONUT'].items():
                if coconut_position < self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                    num_orders = int(min(-vol, delta_hedging_vol, self.POSITION_LIMIT['COCONUT'] - coconut_position))
                    coconut_position += num_orders
                    delta_hedging_vol -= num_orders
                    orders['COCONUT'].append(Order('COCONUT', ask, num_orders))

        elif z_score < -self.coconut_coupon_z_score: # under priced, let's buy coupon sell coconut
            coupon_position = self.position['COCONUT_COUPON']
            fair_buy = best_sell['COCONUT_COUPON']
            coupon_num_orders = 0
            for ask, vol in osell['COCONUT_COUPON'].items():
                if coupon_position < self.POSITION_LIMIT['COCONUT_COUPON'] and (ask <= fair_buy or (coupon_position < 0 and ask == fair_buy + 1)):
                    num_orders = min(-vol, self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                    coupon_position += num_orders
                    coupon_num_orders += num_orders
                    orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', ask, num_orders))
            
            coconut_position, delta_hedging_vol = self.position['COCONUT'], coupon_num_orders * delta
            
            for bid, vol in obuy['COCONUT'].items():
                if coconut_position > -self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                    num_orders = int(max(-vol, -delta_hedging_vol, -self.POSITION_LIMIT['COCONUT'] - coconut_position))
                    coconut_position += num_orders
                    delta_hedging_vol += num_orders
                    orders['COCONUT'].append(Order('COCONUT', bid, num_orders))
                    

        return orders

    def calculate_round4_orders_v2(self, order_depths: dict[str, OrderDepth], data = None)-> dict[str, list]:
        prods = self.round4_products
        orders = {p: [] for p in prods}
        osell, obuy, best_sell, best_buy, mid_price = {}, {}, {}, {}, {}


        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depths[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depths[p].buy_orders.items(), reverse=True))
            
            sell_vol, best_sell_price = self.values_extract(osell[p], buy=False)
            buy_vol, best_buy_price = self.values_extract(obuy[p], buy=True)

            best_sell[p] = best_sell_price
            best_buy[p] = best_buy_price
            mid_price[p] = (best_buy_price + best_sell_price) / 2

        coconut_mid, coupon_mid = mid_price['COCONUT'], mid_price['COCONUT_COUPON']
        
        fair_coupon_price, delta = self.black_scholes_price(S = coconut_mid, delta = True)
        our_bid, our_ask = fair_coupon_price, fair_coupon_price
        logger.print(f" Fair Coupon Price: {fair_coupon_price} Delta: {delta}")


        coupon_position = self.position['COCONUT_COUPON']
        coupon_num_orders = 0
        # MARKET TAKE ASKS (buy items)
        for ask, vol in osell['COCONUT_COUPON'].items():
            if coupon_position < self.POSITION_LIMIT['COCONUT_COUPON']  and (ask <= our_bid or (coupon_position < 0 and ask == our_bid + 1)): 
                num_orders = min(-vol, self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                coupon_position += num_orders
                coupon_num_orders += num_orders
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', ask, num_orders))

        coconut_position, delta_hedging_vol = self.position['COCONUT'], coupon_num_orders * delta
        for bid, vol in obuy['COCONUT'].items():
            if coconut_position > -self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                num_orders = int(max(-vol, -delta_hedging_vol, -self.POSITION_LIMIT['COCONUT'] - coconut_position))
                coconut_position += num_orders
                delta_hedging_vol += num_orders
                orders['COCONUT'].append(Order('COCONUT', bid, num_orders))

        coupon_position = self.position['COCONUT_COUPON']
        coupon_num_orders = 0
        # MARKET TAKE BIDS (sell items)
        for bid, vol in obuy['COCONUT_COUPON'].items():
            if coupon_position > -self.POSITION_LIMIT['COCONUT_COUPON'] and (bid >= our_ask or (coupon_position > 0 and bid == our_ask - 1)):
                num_orders = max(-vol, - self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                coupon_position += num_orders
                coupon_num_orders += num_orders
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', bid, num_orders))

        coconut_position, delta_hedging_vol = self.position['COCONUT'], -coupon_num_orders * delta

        for ask, vol in osell['COCONUT'].items():
            if coconut_position < self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                num_orders = int(min(-vol, delta_hedging_vol, self.POSITION_LIMIT['COCONUT'] - coconut_position))
                coconut_position += num_orders
                delta_hedging_vol -= num_orders
                orders['COCONUT'].append(Order('COCONUT', ask, num_orders))
                

        return orders

    def calculate_round4_orders_v3(self, order_depths: dict[str, OrderDepth], data = None)-> dict[str, list]:
        prods = self.round4_products
        orders = {p: [] for p in prods}
        osell, obuy, best_sell, best_buy, mid_price = {}, {}, {}, {}, {}


        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depths[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depths[p].buy_orders.items(), reverse=True))
            
            sell_vol, best_sell_price = self.values_extract(osell[p], buy=False)
            buy_vol, best_buy_price = self.values_extract(obuy[p], buy=True)

            best_sell[p] = best_sell_price
            best_buy[p] = best_buy_price
            mid_price[p] = (best_buy_price + best_sell_price) / 2

        coconut_mid, coupon_mid = mid_price['COCONUT'], mid_price['COCONUT_COUPON']
        fair_coupon_price, delta = self.black_scholes_price(S = coconut_mid, delta = True)

        implied_vol = self.newtons_method(lambda sigma: self.black_scholes_price(coconut_mid, 10000, 250, 0, sigma) - coupon_mid)
        data.coconut_coupon_implied_vol_cache.append(implied_vol)

        if len(data.coconut_coupon_implied_vol_cache) == data.coconut_coupon_implied_vol_size + 1:
            data.coconut_coupon_implied_vol_cache.pop(0)
            implied_vol_mean, implied_vol_std = np.mean(data.coconut_coupon_implied_vol_cache), np.std(data.coconut_coupon_implied_vol_cache)
            fair_coupon_price, delta = self.black_scholes_price(S = coconut_mid, sigma = implied_vol_mean, delta = True)

        our_bid, our_ask = fair_coupon_price - 1, fair_coupon_price + 1

        coupon_position = self.position['COCONUT_COUPON']
        coupon_num_orders = 0
        # MARKET TAKE ASKS (buy items)
        for ask, vol in osell['COCONUT_COUPON'].items():
            if coupon_position < self.POSITION_LIMIT['COCONUT_COUPON']  and (ask <= our_bid or (coupon_position < 0 and ask == our_bid + 1)): 
                num_orders = min(-vol, self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                coupon_position += num_orders
                coupon_num_orders += num_orders
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', ask, num_orders))

        coconut_position, delta_hedging_vol = self.position['COCONUT'], coupon_num_orders * delta
        for bid, vol in obuy['COCONUT'].items():
            if coconut_position > -self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                num_orders = int(max(-vol, -delta_hedging_vol, -self.POSITION_LIMIT['COCONUT'] - coconut_position))
                coconut_position += num_orders
                delta_hedging_vol += num_orders
                orders['COCONUT'].append(Order('COCONUT', bid, num_orders))

        coupon_position = self.position['COCONUT_COUPON']
        coupon_num_orders = 0
        # MARKET TAKE BIDS (sell items)
        for bid, vol in obuy['COCONUT_COUPON'].items():
            if coupon_position > -self.POSITION_LIMIT['COCONUT_COUPON'] and (bid >= our_ask or (coupon_position > 0 and bid == our_ask - 1)):
                num_orders = max(-vol, - self.POSITION_LIMIT['COCONUT_COUPON'] - coupon_position)
                coupon_position += num_orders
                coupon_num_orders += num_orders
                orders['COCONUT_COUPON'].append(Order('COCONUT_COUPON', bid, num_orders))

        coconut_position, delta_hedging_vol = self.position['COCONUT'], -coupon_num_orders * delta
        for ask, vol in osell['COCONUT'].items():
            if coconut_position < self.POSITION_LIMIT['COCONUT'] and delta_hedging_vol > 0:
                num_orders = int(min(-vol, delta_hedging_vol, self.POSITION_LIMIT['COCONUT'] - coconut_position))
                coconut_position += num_orders
                delta_hedging_vol -= num_orders
                orders['COCONUT'].append(Order('COCONUT', ask, num_orders))

        return orders


    def compute_orders_basket(self, order_depths: dict[str, OrderDepth], data = None)-> dict[str, list]:
        prods = self.round3_products # ['GIFT_BASKET', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES']
        orders = {p: [] for p in prods}
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depths[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depths[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p]) / 2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - self.basket_premium
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - self.basket_premium

        trade_at = self.basket_premium_std * 0.5

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0


        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) # sell
                if res_sell > trade_at * 1.6:
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol * 4))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol * 6))
                    orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                if res_buy < -trade_at * 1.6:
                    orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol * 4))
                    orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol * 6))
                    orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        return orders
    
