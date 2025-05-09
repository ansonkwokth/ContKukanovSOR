import argparse
import logging
from utils.venue import Venue
from utils.data import load_and_clean_l1_data
from utils.helpers import *
from utils.logger import get_logger
from allocator import allocate, compute_cost # assuming allocator is in the same directory

import numpy as np
import matplotlib.pyplot as plt

# Set up the logger
logger = get_logger('backtest', logging.INFO)
# logger = get_logger('backtest', logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Order Router Backtest")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to L1 message CSV file")
    parser.add_argument("--lambda_over", type=float, default=0, help="Overfill penalty")
    parser.add_argument("--lambda_under", type=float, default=500, help="Underfill penalty")
    parser.add_argument("--theta_queue", type=float, default=0, help="Queue-risk penalty")
    parser.add_argument("--order_size", type=int, default=5000, help="Buy order size")
    return parser.parse_args()

def backtest(file: str, order_size_to_fill: int, lambda_over: float, lambda_under: float, theta_queue: float):
    """
    Runs a backtest by reading L1 market data from a CSV, allocating the order,
    and then calculating the cost using the allocator.

    Args:
        file (str): Path to the cleaned L1 data CSV file.
        order_size (int): Total number of shares to be bought in the order.
        lambda_over (float): Cost penalty for extra shares bought.
        lambda_under (float): Cost penalty for unfilled shares.
        theta_queue (float): Queue risk penalty (linear in total mis-execution).
    """
    logger.info(f"Starting backtest for order size {order_size_to_fill}.")

    # Load and clean L1 data
    snapshots = load_and_clean_l1_data(file)

    logger.debug(f"The first snapshot tuple:\n{snapshots[0]}")

    order_size = order_size_to_fill
    total_cost = 0.0
    total_cash = 0.0

    costs = []
    cashs = []
    asks = []

    # Iterate through each snapshot (each timestamp)
    for idx, (ts_event, snapshot) in enumerate(snapshots):

        # Prepare the list of venues for allocation
        venues = []
        for _, row in snapshot.iterrows():
            venue = Venue(
                ask=row["ask_px_00"],
                ask_size=row["ask_sz_00"],
                fee=0.00,       # Adjust as needed
                rebate=0.00    # Adjust as needed
            )
            # TODO: need to be more flexible in the fee
            logger.debug(f"Created {venue}")
            venues.append(venue)

        logger.debug(f"The venues of the first snapshot:\n{venues}")

        # Allocate the order across the venues using the allocator
        best_split, best_cost = allocate(order_size, venues, lambda_over, lambda_under, theta_queue)
        best_split = clip_split_to_remaining(best_split, order_size)
        #if len(best_split) == 0: print("?"); continue

        logger.info(f"Best split at timestamp {ts_event}: {best_split} with cost {best_cost:.4f}")

        filled = sum(best_split)
        cash = cash_spent(best_split, venues, filled, lambda_over, lambda_under, theta_queue)
        order_size -= filled
        total_cash += cash
        total_cost += best_cost

        logger.debug(f"Remaining order: {order_size} | Snapshot cash: {cash:.2f}")

        costs.append(best_cost)
        cashs.append(cash)

        if order_size == 0:
            break


    logger.info(f"Total cost: {total_cost:.4f}")
    logger.info(f"Average cost per share: {total_cash / order_size_to_fill:.4f}")

    return costs


def plot_cum_cost(costs):
    plt.plot(np.cumsum(costs))
    plt.savefig("cumplot.png")



if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Run the backtest with the parsed arguments
    costs = backtest(args.file, args.order_size, args.lambda_over, args.lambda_under, args.theta_queue)
    plot_cum_cost(costs)


