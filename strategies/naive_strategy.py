from typing import List, Tuple
from utils.venue import Venue

def naive_strategy(order_size: int, venues: List[Venue]) -> List[int]:
    """
    A simple strategy that fills the order by always choosing the venue with the best (lowest) ask price.

    Args:
        order_size (int): Total number of shares to be bought.
        venues (List[Venue]): List of venues to pick from, each with an ask price and ask size.

    Returns:
        List[int]: Allocation of shares to each venue.
        float: Total cost of the order.
    """
    # Sort venues by ask price (lowest ask first)
    min_ask = min([v.ask for v in venues])

    allocation = []

    for v in venues:
        if order_size <= 0:
            qty_to_buy = 0
        if v.ask == min_ask:
            qty_to_buy = v.ask_size
        else:
            qty_to_buy = 0

        allocation.append(qty_to_buy)

        # Update remaining order
        order_size -= qty_to_buy


    return allocation

