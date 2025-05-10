import argparse
import logging
from utils.venue import Venue
from utils.data import load_and_clean_l1_data
from utils.helpers import *
from utils.logger import get_logger
from allocator import allocate, compute_cost # assuming allocator is in the same directory

from strategies.naive_strategy import *
from strategies.SOR_strategy import *
from strategies.vwap_strategy import *
from strategies.twap_strategy import *


import numpy as np
import matplotlib.pyplot as plt

# Set up the logger
logger = get_logger('backtest', logging.INFO)
# logger = get_logger('backtest', logging.DEBUG)

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Order Router Backtest")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to L1 message CSV file")
    parser.add_argument("--lambda_over", type=float, default=0, help="Overfill penalty")
    # parser.add_argument("--lambda_under", type=float, default=500, help="Underfill penalty")
    parser.add_argument("--lambda_under", type=float, default=200, help="Underfill penalty")
    parser.add_argument("--theta_queue", type=float, default=0, help="Queue-risk penalty")
    parser.add_argument("--order_size", type=int, default=5000, help="Buy order size")
    parser.add_argument("--fee", type=float, default=0.005, help="Fee")
    parser.add_argument("--rebate", type=float, default=0.0005, help="Rebate")
    parser.add_argument("--optimize", type=str, default=None, help="Optimization")
    return parser.parse_args()









def backtest(file: str, order_size_to_fill: int, lambda_over: float, lambda_under: float, theta_queue: float,
            fee: float, rebate: float,
            baselines: bool, early_stop: bool):
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


    twap_buckets_size = 60
    twap_buckets = det_buckets(snapshots[0][0], snapshots[-1][0], twap_buckets_size)
    bucket_order_size = round(order_size_to_fill / len(twap_buckets))
    twap_buckets_order_size = {bucket: bucket_order_size for bucket in twap_buckets}
    twap_buckets_order_size[twap_buckets[-1]] = 1
    twap_buckets_order_size[twap_buckets[-1]] = order_size_to_fill - bucket_order_size * (len(twap_buckets)-1)


    order_size = order_size_to_fill
    order_size_ns = order_size_to_fill
    order_size_vwap = order_size_to_fill
    order_size_twap = order_size_to_fill

    total_cost = 0.0

    total_cash = 0.0
    total_cash_ns = 0.0
    total_cash_vwap = 0.0

    total_cash_for_avg_price = 0.0
    total_cash_ns_for_avg_price  = 0.0
    total_cash_vwap_for_avg_price  = 0.0

    costs = []
    costs_ns = []
    costs_vw = []
    costs_tw = []

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
                fee=fee,
                rebate=rebate
            )
            # TODO: need to be more flexible in the fee
            logger.debug(f"Created {venue}")
            venues.append(venue)
        logger.debug(f"The venues of the first snapshot:\n{venues}")




        if baselines:
            alloc_ns = naive_strategy(order_size_ns, venues)
            alloc_ns = clip_split_to_remaining(alloc_ns, order_size_ns)
            filled_ns = sum(alloc_ns)
            cash_ns = cash_spent(alloc_ns, venues, filled_ns)
            costs_ns.append(cash_spent(alloc_ns, venues, filled_ns, False))
            order_size_ns -= filled_ns
            total_cash_ns += cash_ns

            alloc_vw = vwap_strategy(order_size_ns, venues)
            alloc_vw = clip_split_to_remaining(alloc_vw, order_size_vwap)
            filled_vw = sum(alloc_vw)
            costs_vw.append(cash_spent(alloc_vw, venues, filled_vw, False))
            cash_vw = cash_spent(alloc_vw, venues, filled_vw)
            order_size_vwap -= filled_vw
            total_cash_vwap += cash_vw

            alloc_tw, twap_buckets_order_size = twap_strategy(twap_buckets_order_size, venues, twap_buckets, twap_buckets_size, ts_event)
            filled_tw = sum(alloc_tw)
            costs_tw.append(cash_spent(alloc_tw, venues, filled_tw, False))

        best_split, best_cost = SOR_strategy(order_size, venues, lambda_over, lambda_under, theta_queue)
        best_split = clip_split_to_remaining(best_split, order_size)
        # logger.info(f"Best split at timestamp {ts_event}: {best_split} with cost {best_cost:.4f}")
        filled = sum(best_split)
        cash = cash_spent(best_split, venues, filled)
        order_size -= filled
        total_cash += cash
        total_cost += best_cost
        logger.debug(f"Remaining order: {order_size} | Snapshot cash: {cash:.2f}")
        if not baselines and early_stop and order_size == 0:
            break

        costs.append(cash_spent(best_split, venues, filled, False))
        cashs.append(cash)



    logger.info(f"Orders: {order_size}, {order_size_ns}, {order_size_vwap}")
    logger.info(f"Total cost: {total_cost:.4f}")
    logger.info(f"Average cost per share: {total_cash / (order_size_to_fill - order_size):.4f}")
    logger.info(f"NS: Average cost per share: {total_cash_ns / (order_size_to_fill - order_size_ns):.4f}")
    logger.info(f"WV: Average cost per share: {total_cash_vwap / (order_size_to_fill - order_size_vwap):.4f}")
    if order_size != 0:
        return float("inf"), float("inf")

    return total_cash / order_size_to_fill, (costs, costs_ns, costs_vw, costs_tw)


def plot_cum_cost(costs):
    plt.plot(np.cumsum(costs))
    plt.savefig("cumplot.png")



if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    lambda_over = args.lambda_over
    lambda_under = args.lambda_under
    theta_queue = args.theta_queue



    if args.optimize and args.optimize.lower() == "grid":
        print("grid")
        lo_rng = [0]
        lu_rng = np.linspace(200, 5000, 50)
        th_rng = [0]
        min_costs = float("inf")
        for lo in lo_rng:
            for lu in lu_rng:
                for th in th_rng:
                    avg_price, costs = backtest(args.file, args.order_size, lo, lu, th,
                                                args.fee, args.rebate, False, True)
                    if min_costs < np.sum(costs):
                        min_costs = np.sum(costs)
                        lambda_over = lo
                        lambda_under = lu
                        theta_queue = th

                    print(lo, lu, th, avg_price)


    # Run the backtest with the parsed arguments
    avg_price, costs_lt = backtest(args.file, args.order_size, lambda_over, lambda_under, theta_queue,
                                args.fee, args.rebate, True, False)
    plot_cum_cost(costs_lt[0])
    plot_cum_cost(costs_lt[1])
    plot_cum_cost(costs_lt[2])
    plot_cum_cost(costs_lt[3])


