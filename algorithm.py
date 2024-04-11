from typing import List
import numpy as np
import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import copy
import collections

class Logger:
    
    local_logs: dict[int, str] = {}

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

        self.local_logs[state.timestamp] = json.dumps({
            "state": state,
            "orders": orders,
            "logs": self.logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True)

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
    total_vol = 0
    best_val = -1
    max_vol = -1

    for ask, vol in order_dict.items():
        if(buy==0):
            vol *= -1
        total_vol += vol
        if total_vol > max_vol:
            max_vol = vol
            best_val = ask
    
    return total_vol, best_val


class Trader:

    logger = Logger()

    POSITION_LIMIT = {
        'AMETHYSTS': 20,
        'STARFRUIT': 20
    }

    position = {
        'AMETHYSTS': 0,
        'STARFRUIT': 0
    
    }

    round1_products = ['AMETHYSTS', 'STARFRUIT']

    starfruit_coefs = {
# "vwap": 1,
# "vwap_change": -0.25076721339661984,
# "imbalance_volume_signed": 0.02594159961108483,
"best_mean": 0.9970645785765161,
"best_mean_-1": 0.0007666006190972529,
"best_mean_-2": 0.0003870777600799704,
"best_mean_-3": 0.00011440834143569041,
    }

    starfruit_cache = {
        # "vwap": None,
        # "vwap_change": None,
        # "imbalance_volume_signed": None,
        "best_mean": None,
        "best_mean_-1": None,
        "best_mean_-2": None,
        "best_mean_-3": None,
    }

    def predict_starfruit(self):
        res = 8.376824143833801

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
            max_for_buy = -1

            undercut_buy = best_buy_price + 1
            undercut_sell = best_sell_price - 1

            # shift price one to be the best bid/sell
            bid_price = min(undercut_buy, fair_bid - 1)
            sell_price = max(undercut_sell, fair_ask + 1)

            # Search the orders that satisfy our acceptable price
            # ask
            for ask, vol in order_sell.items():
                if ((ask < fair_bid) or ((self.position[product] < 0) and (ask == fair_ask))) and curr_pos < self.POSITION_LIMIT[product]:
                    max_for_buy = max(max_for_buy, ask)
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

            if curr_pos > -LIMIT:
                num = -LIMIT-curr_pos
                orders.append(Order(product, sell_pr, num))
                curr_pos += num

            return orders
        

    
    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent

        result = {
            'AMETHYSTS': [],
            'STARFRUIT': []
        }


        for key, val in state.position.items():
            self.position[key] = val

        timestamp = state.timestamp

        starfruit_lb, starfruit_ub = -1e9, 1e9

        '''
        "vwap": None,
        "vwap_change": None,
        "imbalance_volume_signed": None,
        "best_mean": None,
        "best_mean_-1": None,
        "best_mean_-2": None,
        '''
        
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


        if not any(value is None for value in self.starfruit_cache.values()):
            starfruit_fair_price = self.predict_starfruit()
            starfruit_lb = starfruit_fair_price - 1
            starfruit_ub = starfruit_fair_price + 1
        
        acceptable_bids = {
            'AMETHYSTS': 10000,
            'STARFRUIT': starfruit_lb
        }

        acceptable_asks = {
            'AMETHYSTS': 10000,
            'STARFRUIT': starfruit_ub
        }

        for product in self.round1_products:
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.calc_round1_order(product, order_depth, acceptable_bids[product], acceptable_asks[product],)
            if orders:
                result[product] += orders
    
        traderData = "" # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        
        conversions = 1

        self.logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData