from typing import List, Tuple
from utils.venue import Venue
import pandas as pd
from datetime import timedelta

def det_buckets(start: str, end: str, size: float = 60):
    """
    Generate a list of bucket start times between `start` and `end`,
    spaced by `size` seconds. Handles nanosecond ISO 8601 strings.
    Args:
        start (str): ISO8601 timestamp with nanoseconds
        end (str): ISO8601 timestamp with nanoseconds
        size (int): bucket size in seconds (default = 60)
    Returns:
        List of pandas.Timestamp bucket start times
    """
    start_time = pd.to_datetime(start)
    end_time = pd.to_datetime(end)
    bucket_size = timedelta(seconds=size)

    buckets = []
    current = start_time
    while current < end_time:
        buckets.append(current)
        current += bucket_size

    return buckets




def twap_strategy(bkt_order_size: int, venues: List[Venue]) -> Tuple[List[int], int]:
    """
    TWAP (Time-Weighted Average Price) strategy that allocates a bucketed order size
    across multiple venues based on their ask prices and available sizes.

    Args:
        bkt_order_size (int): The total size of the order to be executed.
        venues (List[Venue]): A list of venue objects, each with an 'ask' price and 'ask_size'.

    Returns:
        Tuple[List[int],int]:
            - A list of ints representing the amount of order allocated to each venue.
            - The remaining unallocated portion of the order size (if any).
    """

    allocation = [0] * len(venues)

    total_size = sum(venue.ask_size for venue in venues)
    if total_size == 0:
        return allocation, bkt_order_size
    if bkt_order_size == 0:
        return allocation, bkt_order_size

    sorted_indices = [i for i, _ in sorted(enumerate(venues), key=lambda x: x[1].ask)]

    for idx in sorted_indices:
        v = venues[idx]
        exe_size = min(v.ask_size, bkt_order_size)
        allocation[idx] = exe_size
        bkt_order_size -= exe_size

    return allocation, bkt_order_size







