import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from allocator import *


def SOR_strategy(order_size, venues, lambda_over, lambda_under, theta_queue):
    if order_size <= 0:
        return [0] * len(venues), 0
    # Allocate the order across the venues using the allocator
    best_split, best_cost = allocate(order_size, venues, lambda_over, lambda_under, theta_queue)
    if len(best_split) == 0:
        return [0] * len(venues), 0

    return best_split, best_cost
