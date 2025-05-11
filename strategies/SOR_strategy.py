from typing import List, Tuple
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from strategies.allocator import *


def SOR_strategy(
        order_size: int,
        venues: List[Venue],
        lambda_over: float,
        lambda_under: float,
        theta_queue: float
        ) -> Tuple[List[int], float]:

    if order_size <= 0:
        return [0] * len(venues), 0.0

    # Allocate the order across the venues using the allocator
    best_split, best_cost = allocate(order_size, venues, lambda_over, lambda_under, theta_queue)

    if not best_split:
        return [0] * len(venues), 0.0

    return best_split, best_cost
