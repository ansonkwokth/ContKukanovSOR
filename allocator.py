from typing import List, Tuple
from utils.logger import get_logger
import logging

logger = get_logger("Allocator")
# logger = get_logger("Allocator", log_level=logging.DEBUG)

class Venue:
    """
    A class representing a trading venue.
    Attributes:
        ask (float): The price of the best ask on the venue.
        ask_size (int): The size of the best ask.
        fee (float): The fee charged by the venue for a trade.
        rebate (float): The rebate provided by the venue for market makers.
    """
    def __init__(self, ask: float, ask_size: int, fee: float, rebate: float):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate



def compute_cost(split: List[int],
                 venues: List[Venue],
                 order_size: int,
                 lambda_over: float,
                 lambda_under: float,
                 theta_queue: float) -> float:
    """
    Computes the total cost for a given allocation (split) across venues.

    Args:
        split (List[int]): A list of integers representing how many shares are allocated to each venue.
        venues (List[Venue]): A list of Venue objects representing the available venues.
        order_size (int): The total order size to be executed.
        lambda_over (float): Cost penalty per extra share bought (overfill).
        lambda_under (float): Cost penalty per unfilled share (underfill).
        theta_queue (float): Queue-risk penalty.

    Returns:
        float: The total cost for the given allocation.
    """
    executed = 0
    cash_spent = 0.0

    for i in range(len(venues)):
        exe = min(split[i], venues[i].ask_size)
        executed += exe
        cash_spent += exe * (venues[i].ask + venues[i].fee)
        maker_rebate = max(split[i] - exe, 0) * venues[i].rebate
        cash_spent -= maker_rebate

    underfill = max(order_size - executed, 0)
    overfill = max(executed - order_size, 0)
    risk_pen = theta_queue * (underfill + overfill)
    cost_pen = lambda_under * underfill + lambda_over * overfill

    total_cost = cash_spent + risk_pen + cost_pen
    logger.debug(f"Split: {split} | Exec: {executed} | Cost: {total_cost:.2f}")
    return total_cost



def allocate(order_size: int,
             venues: List[Venue],
             lambda_over: float,
             lambda_under: float,
             theta_queue: float) -> Tuple[List[int], float]:
    """
    Allocates an order across multiple venues in a way that minimizes the expected cost,
    following the Cont & Kukanov static model.

    Args:
        order_size (int): The total size of the order to be split across venues.
        venues (List[Venue]): A list of Venue objects representing the available venues.
        lambda_over (float): Cost penalty per extra share bought (overfill).
        lambda_under (float): Cost penalty per unfilled share (underfill).
        theta_queue (float): Queue-risk penalty (linear in total mis-execution).

    Returns:
        Tuple[List[int], float]: A tuple containing:
            - A list of integers, each representing the number of shares allocated to each venue.
            - The total cost for the chosen allocation.
    """
    step = 100
    splits = [[]]
    logger.info("Starting allocation search...")

    for v in range(len(venues)):
        new_splits = []
        for alloc in splits:
            used = sum(alloc)
            max_v = min(order_size - used, venues[v].ask_size)
            for q in range(0, max_v + 1, step):
                new_splits.append(alloc + [q])
        splits = new_splits
        logger.debug(f"Venue {v}: {len(splits)} candidate splits")

    best_cost = float("inf")
    best_split = []

    for alloc in splits:
        if sum(alloc) != order_size: continue
        cost = compute_cost(alloc, venues, order_size,
                            lambda_over, lambda_under, theta_queue)
        if cost < best_cost:
            best_cost = cost
            best_split = alloc

    logger.info(f"Best split found: {best_split} with cost {best_cost:.2f}")
    return best_split, best_cost



# Quick test
if __name__ == "__main__":
    venues = [
        Venue(ask=10.0, ask_size=3000, fee=0.01, rebate=0.002),
        Venue(ask=10.1, ask_size=3000, fee=0.01, rebate=0.002)
    ]
    order_size = 5000
    split, cost = allocate(order_size, venues, lambda_over=0.1, lambda_under=0.1, theta_queue=0.05)
    print("Best Split:", split)
    print("Expected Cost:", round(cost, 2))

