from typing import List
import numpy as np
import json
from datamodel import ConversionObservation, Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import copy
import collections


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

def calc_vwap(order_depth):

    buy_price = np.array([item for item in order_depth.buy_orders.keys() if not np.isnan(item)])

    buy_volume = np.array([item for item in order_depth.buy_orders.values() if not np.isnan(item)])

    sell_price = np.array([item for item in order_depth.sell_orders.keys() if not np.isnan(item)])

    sell_volume = -1 * np.array([item for item in order_depth.sell_orders.values() if not np.isnan(item)])

    '''
    buy_wvap = float((buy_price * buy_volume).sum() / buy_volume.sum())

    sell_wvap = float((sell_price * sell_volume).sum() / sell_volume.sum())
    '''

    vwap = float(((buy_price * buy_volume).sum() + (sell_price * sell_volume).sum()) / (buy_volume.sum() + sell_volume.sum()))

    return vwap


def calculate_imbalance(order_depth):
    
    buy_volume = np.array([item for item in order_depth.buy_orders.values() if not np.isnan(item)]).sum()

    sell_volume = np.array([item for item in order_depth.sell_orders.values() if not np.isnan(item)]).sum()

    # Calculate and return the VWAP
    return (buy_volume - sell_volume)


def values_extract(order_dict, buy=0):
    volume = 0
    best = 0 if buy else float('inf')

    for price, vol in order_dict.items():
        if buy:
            volume += vol
            best = max(best, price)
        else:
            volume -= vol
            best = min(best, price)

    return volume, best


class Trader:


    POSITION_LIMIT = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20,
        'ORCHIDS': 100,
        'CHOCOLATE': 250,
        'STRAWBERRIES': 350,
        'ROSES': 60,
        'GIFT_BASKET': 60,
    }

    position = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0,
        'ORCHIDS': 0,
        'CHOCOLATE': 0,
        'STRAWBERRIES': 0,
        'ROSES': 0,
        'GIFT_BASKET': 0,
    }

    round1_products = ['AMETHYSTS', 'STARFRUIT']
    round2_products = ['ORCHIDS']

    starfruit_coefs = {
        # "vwap": 1,
        # "vwap_change": -0.25076721339661984,
        # "imbalance_volume_signed": -0.17395201352494163,
"best_mean": 1.0012252201206766,
"best_mean_-1": -0.00011613099877059405,
"best_mean_-2": 0.0003941628661304437,
"best_mean_-3": -0.00023525537272049696,
        # "best_mean_MA_3": -0.4522749419653542,
        # "best_mean_MA_10": 0.00012514476494644223,
    }
    starfruit_best_mean = []

    starfruit_cache = {
        # "vwap": None,
        # "vwap_change": None,
        # "imbalance_volume_signed": None,
        "best_mean": None,
        "best_mean_-1": None,
        "best_mean_-2": None,
        "best_mean_-3": None,
        # "best_mean_MA_3": None,
        # "best_mean_MA_10": None,
    }


    cumulative_sunlight = 0
    past_humidity = []
    humidity_change = []
    


    def predict_starfruit(self):
        res = -6.523275510305211

        for key, val in self.starfruit_cache.items():
            res += val * self.starfruit_coefs[key]

        return int(round(res))

    def calc_round1_order(self, product: str, order_depth: OrderDepth, 
                          fair_bid: float, fair_ask: float) -> List[Order]:
        if product == 'AMETHYSTS':
            orders = []
            
            order_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            order_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

            sell_vol, best_sell_price = values_extract(order_sell)
            buy_vol, best_buy_price = values_extract(order_buy, 1)

            curr_pos = self.position[product]

            undercut_buy = best_buy_price + 1
            undercut_sell = best_sell_price - 1

            # shift price one to be the best bid/sell
            bid_price = min(undercut_buy, fair_bid - 1)
            sell_price = max(undercut_sell, fair_ask + 1)

            # Search the orders that satisfy our acceptable price
            # ask
            for ask, vol in order_sell.items():
                if ((ask < fair_bid) or ((self.position[product] < 0) and (ask == fair_ask))) and curr_pos < self.POSITION_LIMIT[product]:
                    order_for = min(-vol, self.POSITION_LIMIT[product])
                    curr_pos += order_for
                    orders.append(Order(product, ask, order_for))
            
            if (curr_pos < self.POSITION_LIMIT[product]) and (self.position[product] < 0):
                num = min(40, self.POSITION_LIMIT[product] - curr_pos)
                orders.append(Order(product, min(undercut_buy + 1, fair_bid - 1), num))
                curr_pos += num

            if (curr_pos < self.POSITION_LIMIT[product]) and (self.position[product] > 15):
                num = min(40, self.POSITION_LIMIT[product] - curr_pos)
                orders.append(Order(product, min(undercut_buy - 1, fair_bid - 1), num))
                curr_pos += num

            if curr_pos < self.POSITION_LIMIT[product]:
                num = min(40, self.POSITION_LIMIT[product] - curr_pos)
                orders.append(Order(product, bid_price, num))
                curr_pos += num

            # bid
            curr_pos = self.position[product]

            for bid, vol in order_buy.items():
                if ((bid > fair_ask) or ((self.position[product]>0) and (bid == fair_ask))) and curr_pos > -self.POSITION_LIMIT[product]:
                    order_for = max(-vol, -self.POSITION_LIMIT[product]-curr_pos)
                    # order_for is a negative number denoting how much we will sell
                    curr_pos += order_for
                    assert(order_for <= 0)
                    orders.append(Order(product, bid, order_for))

            if (curr_pos > -self.POSITION_LIMIT[product]) and (self.position[product] > 0):
                num = max(-40, -self.POSITION_LIMIT[product]-curr_pos)
                orders.append(Order(product, max(undercut_sell - 1, fair_ask + 1), num))
                curr_pos += num

            if (curr_pos > -self.POSITION_LIMIT[product]) and (self.position[product] < -15):
                num = max(-40, -self.POSITION_LIMIT[product]-curr_pos)
                orders.append(Order(product, max(undercut_sell + 1, fair_ask + 1), num))
                curr_pos += num

            if curr_pos > -self.POSITION_LIMIT[product]:
                num = max(-40, -self.POSITION_LIMIT[product] - curr_pos)
                orders.append(Order(product, sell_price, num))
                curr_pos += num

            return orders
        
        elif product == 'STARFRUIT':
            LIMIT = self.POSITION_LIMIT[product]
            orders: list[Order] = []

            order_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
            order_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

            sell_vol, best_sell_price = values_extract(order_sell)
            buy_vol, best_buy_price = values_extract(order_buy, 1)

            best_buy_price = list(order_buy.items())[0][0]
            best_sell_price = list(order_sell.items())[0][0]

            curr_pos = self.position[product]

            for ask, vol in order_sell.items():
                if ((ask <= fair_bid) or ((self.position[product] < 0) and (ask == fair_bid + 1))) and curr_pos < LIMIT:
                    order_for = min(-vol, LIMIT - curr_pos)
                    curr_pos += order_for
                    assert(order_for >= 0)
                    orders.append(Order(product, ask, order_for))

            undercut_buy = best_buy_price + 1
            undercut_sell = best_sell_price - 1

            bid_pr = min(undercut_buy, fair_bid) # we will shift this by 1 to beat this price
            sell_pr = max(undercut_sell, fair_ask)

            if curr_pos < LIMIT:
                num = LIMIT - curr_pos
                orders.append(Order(product, bid_pr, num))
                curr_pos += num
            
            curr_pos = self.position[product]

            for bid, vol in order_buy.items():
                if ((bid >= fair_ask) or ((self.position[product] > 0) and (bid+1 == fair_ask))) and curr_pos > -LIMIT:
                    order_for = max(-vol, -LIMIT-curr_pos)
                    # order_for is a negative number denoting how much we will sell
                    curr_pos += order_for
                    assert(order_for <= 0)
                    orders.append(Order(product, bid, order_for))

            return orders
        
    def calc_round2_order(self, product: str, order_depth: OrderDepth, 
                          fair_bid: float, fair_ask: float, orchids_observation: ConversionObservation, timestamp: int = 0) -> List[Order]:
        
        orders: list[Order] = []
        
        sell_orders = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        buy_orders = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_price = values_extract(sell_orders, buy=0)
        buy_vol, best_buy_price = values_extract(buy_orders, buy=1)

        position = 0
        LIMIT = self.POSITION_LIMIT[product]
        MM_spread = 5

        # penny the current highest bid / lowest ask 
        fair_local_buy = best_buy_price + 1
        fair_local_sell = best_sell_price - 1

        real_bid = orchids_observation.bidPrice - orchids_observation.exportTariff - orchids_observation.transportFees - 0.1
        real_ask = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees
        
        our_bid = int(real_ask) - 1
        our_ask = int(real_ask) + 1

        bid_price = min(fair_local_buy, our_bid)
        ask_price = max(fair_local_sell, our_ask, best_sell_price - MM_spread)


        logger.print(f"##{sell_orders} $$ {buy_orders} #####real_bid {real_bid}, real_ask {real_ask}, our_bid {our_bid},  our_ask {our_ask}  bid_price {bid_price} ask_price {ask_price} ")

        # MARKET TAKE ASKS (buy items)
        for ask, vol in sell_orders.items():
            if position < LIMIT and (ask <= our_bid or (position < 0 and ask == our_bid+1)): 
                num_orders = min(-vol, LIMIT - position)
                position += num_orders
                orders.append(Order(product, ask, num_orders))

        # MARKET MAKE BY PENNYING
        if position < LIMIT:
            num_orders = LIMIT - position
            orders.append(Order(product, bid_price, num_orders))
            position += num_orders

        # RESET POSITION
        position = 0

        # MARKET TAKE BIDS (sell items)
        for bid, vol in buy_orders.items():
            if position > -LIMIT and (bid >= our_ask or (position > 0 and bid+1 == our_ask)):
                num_orders = max(-vol, -LIMIT-position)
                position += num_orders
                orders.append(Order(product, bid, num_orders))

        # MARKET MAKE BY PENNYING
        if position > -LIMIT:
            num_orders = -LIMIT - position
            orders.append(Order(product, ask_price, num_orders))
            position += num_orders 


        logger.print(f"placed orders: {orders}")
        return orders

        
        """
        orders: list[Order] = []
        conversions: int = 0
        
        LIMIT = self.POSITION_LIMIT[product]

        order_sell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        order_buy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        
        real_bid = orchids_observation.bidPrice - orchids_observation.exportTariff - orchids_observation.transportFees - 0.1
        real_ask = orchids_observation.askPrice + orchids_observation.importTariff + orchids_observation.transportFees
        
        print(f"real bid {real_bid}  real ask {real_ask}  current_pos: {self.position[product]}")

        print(order_sell, order_buy)

        curr_pos = self.position[product]
        
        # Implement Arbitrage just by import/export First
        
        for ask, vol in order_sell.items():
            if (ask < real_bid):
                order_for = min(-vol, LIMIT-curr_pos)
                curr_pos += order_for # the order is balanced by conversion
                orders.append(Order(product, ask, order_for))
                orders.append(Order(product, int(real_bid), -order_for))
                conversions -= order_for
        
        for bid, vol in order_buy.items():
            if (bid > real_ask):
                order_for = max(-vol, -LIMIT-curr_pos)
                curr_pos += order_for # the order is balanced by conversion
                orders.append(Order(product, bid, order_for))
                orders.append(Order(product, int(real_ask + 0.9), -order_for))
                conversions -= order_for


        sell_vol, best_sell_price = values_extract(order_sell)
        buy_vol, best_buy_price = values_extract(order_buy, 1)

        best_sell_price = list(order_sell.items())[0][0]
        best_buy_price = list(order_buy.items())[0][0]
        sell_vol, buy_vol = list(order_sell.items())[0][1], list(order_buy.items())[0][1]
        """
        '''
        spread = int((best_sell_price - best_buy_price) / 2)
        fair_ask = int((best_sell_price + best_buy_price) / 2 + 0.5)# best_buy_price + spread
        if fair_ask > int(real_ask + 0.9): # if the local ask - 3 > real_ask in south, we can buy real ask and sell ask - 3 on local
            order_for = max(-min(sell_vol, buy_vol), -LIMIT - curr_pos)# -int(0.9 * (LIMIT - curr_pos)) # don't care for position, we can balance by conversion
            orders.append(Order(product, fair_ask, order_for))
            orders.append(Order(product, int(real_ask + 0.9), -order_for))
            conversions -= order_for
            print(f" Want to sell on local with the price {fair_ask} in {order_for} shares and would buy with the price {int(real_ask + 0.9)} in {-order_for} shares")
        
        
        if (best_buy_price + spread) < real_bid: # if the local ask - 3 > real_ask in south, we can buy real ask and sell ask - 3 on local
            order_for = int(0.6 * (LIMIT - curr_pos)) # don't care for position, we can balance by conversion
            orders.append(Order(product, best_buy_price + spread, order_for))
            orders.append(Order(product, int(real_bid), -order_for))
            conversions -= order_for
            print(f" Want to buy on local with the price {best_buy_price + spread} in {order_for} shares and would buy with the price {int(real_bid)} in {-order_for} shares")
        
         '''
        """
        if (best_sell_price + best_buy_price) / 2 > real_ask and (best_sell_price - best_buy_price) > 2:
            order_for = max(-min(-sell_vol, buy_vol), - LIMIT - curr_pos)
            curr_pos += order_for
            orders.append(Order(product, int(real_ask + 2), order_for))
            orders.append(Order(product, int(real_ask + 0.9), -order_for))
            print(f" Want to sell on local with the price {int(real_ask + 2)} in {order_for} shares and would buy with the price {int(real_ask + 0.9)} in {-order_for} shares")
            conversions -= order_for
       
        """
        '''
        Arbitrage from sunlight & humidity 

        sunlight = orchids_observation.sunlight
        humidity = orchids_observation.humidity

        self.cumulative_sunlight += sunlight * 12 / 10000 # 10000 datapoints per trading day; per trading day 12 hours
        remaining_time = (10000 - timestamp // 100) * 12 / 10000
        # expected sunlight: 7*2500, or 420 minutes sunlight with 2500 / hour
        expected_sunlight = 7 * 2500
        forecast_sunlight = self.cumulative_sunlight + remaining_time * sunlight

        sunlight_pos = 0
        if forecast_sunlight < expected_sunlight: # the production decreasing, long now
            minutes_less = (expected_sunlight - forecast_sunlight) / 2500 * 60
            sunlight_pos += minutes_less / 10 * 4
        
        humidity_pos = 0
        if not self.past_humidity:
            self.past_humidity.append(humidity)
        elif len(self.past_humidity) < 200:
            self.past_humidity.append(humidity)
            self.humidity_change.append(humidity - self.past_humidity[-1])
        else:
            self.past_humidity.pop(0)
            self.humidity_change.pop(0)
            self.past_humidity.append(humidity)
            self.humidity_change.append(humidity - self.past_humidity[-1])

        if 60 <= humidity <= 80:
            if sunlight_pos != 0:
                order_for = min(-sell_vol, sunlight_pos, LIMIT - curr_pos)
                orders.append(Order(product, best_sell_price + 1, order_for))
                curr_pos += order_for
                print(f"Orders pure sunlight {best_sell_price + 1}  {order_for}")
        elif humidity < 60:
            humidity_diff = 60 - humidity
            production_decrease = humidity_diff * 0.4
            if self.past_humidity == 200:
                if sum(self.humidity_change) > 0: # go back to the range, we should short
                    humidity_pos -= production_decrease
                elif sum(self.humidity_change) < 0: # still out of the range, we should long
                    humidity_pos += production_decrease
        elif humidity > 80:
            humidity_diff = humidity - 80
            production_decrease = humidity_diff * 0.4
            if self.past_humidity == 200:
                if sum(self.humidity_change) < 0: # go back to the range, we should short
                    humidity_pos -= production_decrease
                elif sum(self.humidity_change) > 0: # still out of the range, we should long
                    humidity_pos += production_decrease
            
                
        change_pct = (100 + sunlight_pos) * (100 + humidity_pos) / 10000
        fair_price = (best_sell_price + best_buy_price) // 2

        if change_pct > 1:
            order_for = LIMIT - curr_pos   
            orders.append(Order(product, fair_price, order_for))
            print(f"Orders combined {fair_price}  {order_for}")
            curr_pos += order_for
        elif change_pct < 1:
            order_for = -LIMIT - curr_pos
            orders.append(Order(product, fair_price, order_for))
            print(f"Orders combined {fair_price}  {order_for}")
            curr_pos += order_for
        
        print(str(orders), conversions)
        return orders, conversions
        '''


    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent

        result = {
            'AMETHYSTS': [],
            'STARFRUIT': [],
            'ORCHIDS':[],
        }


        for key, val in state.position.items():
            self.position[key] = val

        timestamp = state.timestamp // 100

        # ROUND 1
        starfruit_lb, starfruit_ub = -1e9, 1e9
        '''
        starfruit_pre_vwap = self.starfruit_cache["vwap"] 
        self.starfruit_cache["vwap"] = calc_vwap(order_depth = state.order_depths['STARFRUIT'])
        if starfruit_pre_vwap:
            self.starfruit_cache["vwap_change"] = self.starfruit_cache["vwap"] - starfruit_pre_vwap
        
        self.starfruit_cache['imbalance_volume_signed'] = calculate_imbalance(order_depth = state.order_depths['STARFRUIT'])
        '''

        self.starfruit_cache['best_mean_-3'] = self.starfruit_cache['best_mean_-2']
        self.starfruit_cache['best_mean_-2'] = self.starfruit_cache['best_mean_-1']
        self.starfruit_cache['best_mean_-1'] = self.starfruit_cache['best_mean']
        _, best_sell_starfruit = values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, best_buy_starfruit = values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse= True)), 1)
        
        self.starfruit_cache['best_mean'] = (best_buy_starfruit + best_sell_starfruit) / 2
        '''
        if len(self.starfruit_best_mean) == 10:
            self.starfruit_best_mean.pop(0)
            self.starfruit_best_mean.append(self.starfruit_cache['best_mean'])
            self.starfruit_cache[ "best_mean_MA_10"] = sum(self.starfruit_best_mean) / 10
            self.starfruit_cache[ "best_mean_MA_3"] = sum(self.starfruit_best_mean[-3:]) /3
        else:
            self.starfruit_best_mean.append(self.starfruit_cache['best_mean'])
        '''

        if not any(value is None for value in self.starfruit_cache.values()):
            starfruit_fair_price = self.predict_starfruit()
            starfruit_lb = starfruit_fair_price - 1
            starfruit_ub = starfruit_fair_price + 1
        
        acceptable_bids = {
            'AMETHYSTS': 10000,
            'STARFRUIT': starfruit_lb,
            'ORCHIDS': None,
        }

        acceptable_asks = {
            'AMETHYSTS': 10000,
            'STARFRUIT': starfruit_ub,
            'ORCHIDS': None,
        }

        for product in self.round1_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.calc_round1_order(product, order_depth, acceptable_bids[product], acceptable_asks[product],)
            if orders:
                result[product] += orders

        '''
        # ROUND 2
        conversions = 0
        orchids_observation = state.observations.conversionObservations['ORCHIDS']


        for product in self.round2_products:
            order_depth: OrderDepth = state.order_depths[product]
            logger.print(f"observation -- bid:{orchids_observation.bidPrice},  ask:{orchids_observation.askPrice}, export:{orchids_observation.exportTariff},  import:{orchids_observation.importTariff}, transport;{orchids_observation.transportFees}")
            orders = self.calc_round2_order(product, order_depth, acceptable_bids[product], acceptable_asks[product], orchids_observation, timestamp)
            if orders:
                result[product] += orders
            conversions -= self.position[product]
        '''
        traderData = "" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        

        conversions = 0
        # logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData