from typing import List, Tuple
from utils.venue import Venue

def vwap_strategy(order_size: int, venues: List[Venue]) -> List[int]:
    """
    VWAP (Volume-Weighted Average Price) strategy that allocates a given order size
    across multiple venues based on the available ask sizes at each venue.

    Args:
        order_size (int): The total size of the order to be executed.
        venues (List[Venue]): A list of venue objects, each with an 'ask_size' attribute
                               representing the available size at that venue.

    Returns:
        List[int]: A list of allocated order sizes corresponding to each venue.
    """
    total_size = sum(venue.ask_size for venue in venues)

    if total_size == 0:
        return [0] * len(venues)

    allocation = [venue.ask_size for venue in venues]


    return allocation
