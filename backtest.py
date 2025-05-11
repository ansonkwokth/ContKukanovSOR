import json
import yaml
from typing import List
import argparse
import logging
from utils.venue import Venue
from utils.data import load_and_clean_l1_data
from utils.helpers import *
from utils.logger import get_logger

from strategies.naive_strategy import *
from strategies.SOR_strategy import *
from strategies.allocator import allocate, compute_cost # assuming allocator is in the same directory
from strategies.vwap_strategy import *
from strategies.twap_strategy import *


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import datetime

# Set up the logger
logger = get_logger('backtest', logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Smart Order Router Backtest")
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to L1 message CSV file")
    parser.add_argument("--lambda_over", type=float, default=10, help="Overfill penalty")
    parser.add_argument("--lambda_under", type=float, default=20, help="Underfill penalty")
    parser.add_argument("--theta_queue", type=float, default=0, help="Queue-risk penalty")
    parser.add_argument("--order_size", type=int, default=5000, help="Buy order size")
    parser.add_argument("--fee", type=float, default=0.005, help="Fee")
    parser.add_argument("--rebate", type=float, default=0.0005, help="Rebate")
    parser.add_argument("--optimize_config", type=str, default=None, help="Path to optimization configuration file")
    parser.add_argument("--early_stop", type=str, default=False, help="Stop when C & K filled all orders")
    parser.add_argument("--plot_path", type=str, default='result.pdf', help="Path to the output plot")
    return parser.parse_args()





def backtest(snapshots: list[pd.DataFrame], order_size_to_fill: int,
            lambda_over: float, lambda_under: float, theta_queue: float,
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
        fee (float): Fee (assume to be the same across venues here, can be improved using a config file)
        rebate (float): Rebate (assume to be the same across venues here, can be improved using a config file)
    """

    # logger.info(f"Starting backtest for order size {order_size_to_fill}.")


    # logger.debug(f"The first snapshot tuple:\n{snapshots[0]}")

    # TWAP prepare work
    bkts_size = 60  # 60s bucket size (can be made a parameter)
    # Determine bucket boundaries from snapshot timestamps
    buckets = det_buckets(snapshots[0][0], snapshots[-1][0], bkts_size)
    # Compute order size per bucket
    bkt_order_size = round(order_size_to_fill / len(buckets))
    # Initialize current bucket order size
    bkti_order_size = bkt_order_size
    # Start with the first bucket
    bkt_start = buckets.pop(0)

    # For counting the remaining order
    order_size = order_size_to_fill
    # For calculating the average price
    total_price = 0.0
    # For plotting the cumulative cost
    costs = []

    # Baselines parameters
    order_size_bls = {
            "BestAsk": order_size_to_fill,
            "VWAP": order_size_to_fill,
            "TWAP": order_size_to_fill
            }
    total_price_bls = {
            "BestAsk": 0.0,
            "VWAP": 0.0,
            "TWAP": 0.0
            }
    costs_bls = {
            "BestAsk": [],
            "VWAP": [],
            "TWAP": []
            }


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
            # logger.debug(f"Created {venue}")
            venues.append(venue)
        # logger.debug(f"The venues of the first snapshot:\n{venues}")


        # =================== SOR Straetgy ========================
        best_split, best_cost = SOR_strategy(order_size, venues, lambda_over, lambda_under, theta_queue)
        # in case of multiple venues case the the order_size < sum(ask_size).
        # then will just fill order_size
        best_split = clip_split_to_remaining(best_split, order_size)
        # logger.info(f"Best split at timestamp {ts_event}: {best_split} with cost {best_cost:.4f}")
        filled = sum(best_split)
        # cash spent: not the same as the 'best_cost', as it includes both explicit and inplicit costs
        cost = cash_spent(best_split, venues, filled)
        costs.append(cost)
        order_size -= filled
        # logger.debug(f"Remaining order: {order_size} | Snapshot cash: {cash:.2f}")
        total_price += cash_spent(best_split, venues, filled, False)


        # If compare to the baslines
        # Turn this off when tuning the hyperparameters to save time
        if baselines:
            # Naive Best Ask 
            alloc_ns = naive_strategy(order_size_bls["BestAsk"], venues)
            alloc_ns = clip_split_to_remaining(alloc_ns, order_size_bls["BestAsk"])
            filled_ns = sum(alloc_ns)
            cost_ns = cash_spent(alloc_ns, venues, filled_ns)
            costs_bls["BestAsk"].append(cost_ns)
            order_size_bls["BestAsk"] -= filled_ns
            total_price_bls["BestAsk"] += cash_spent(alloc_ns, venues, filled_ns, False)

            # VWAP
            alloc_vw = vwap_strategy(order_size_bls["VWAP"], venues)
            alloc_vw = clip_split_to_remaining(alloc_vw, order_size_bls["VWAP"])
            filled_vw = sum(alloc_vw)
            cost_vw = cash_spent(alloc_vw, venues, filled_vw)
            costs_bls["VWAP"].append(cost_vw)
            order_size_bls["VWAP"] -= filled_vw
            total_price_bls["VWAP"] += cash_spent(alloc_vw, venues, filled_vw, False)

            # TWAP
            # First check which is the current bucket
            ts_time = pd.to_datetime(ts_event)
            # If the time jumps to the next one, update the bucket
            if ts_time > bkt_start + timedelta(seconds=bkts_size):
                bkti_order_size = bkt_order_size
                bkt_start = buckets.pop(0)
            # If all orders have been filled, then bucket doesn't need to fill anything
            if order_size_bls["TWAP"] == 0:
                bkti_order_size = 0
            alloc_tw, bkti_order_size = twap_strategy(bkti_order_size, venues)
            filled_tw = sum(alloc_tw)
            cost_tw = cash_spent(alloc_tw, venues, filled_tw)
            costs_bls["TWAP"].append(cost_tw)
            order_size_bls["TWAP"] -= filled_tw
            total_price_bls["TWAP"] += cash_spent(alloc_tw, venues, filled_tw, False)

        # For optimizing only (to save time)
        # Can already stop the backtest when all orders are filled
        if early_stop and order_size == 0:
            break


    if order_size != 0:
        return (costs, costs_bls), (float('inf'), {}), (order_size == 0)


    avg_price = total_price/ (order_size_to_fill - order_size)
    avg_price_bls = {k: total_price_bls[k]/ (order_size_to_fill - order_size_bls[k]) for k in total_price_bls.keys()} if baselines else {}
    #logger.info(f"Orders: {order_size}")
    #logger.info(f"Total cost: {sum(costs):.4f}")
    #logger.info(f"Average price: {avg_price:.4f}")
    #if baselines: logger.info(f"[BestAsk] Average price: {avg_price_bls['BestAsk']:.4f}")
    #if baselines: logger.info(f"[VWAP] Average price: {avg_price_bls['VWAP']:.4f}")
    #if baselines: logger.info(f"[TWAP] Average price: {avg_price_bls['TWAP']:.4f}")

    return (costs, costs_bls), (avg_price, avg_price_bls), (order_size == 0)


def plot_cost(snapshots, costs: float, costs_bls: List[float], params: dict, out: str):
    fig, ax = plt.subplots(figsize=(12,8))
    tss = range(len(costs))
    #tss = [snapshot[0] for snapshot in snapshots][:len(costs)]
    #tss = pd.to_datetime(tss, utc=True).strftime("%H:%M:%S.%f")
    ax.set_title(r"$\lambda_{\text{over}}$="+f"{params['lambda_over']};    "+\
            r"$\lambda_{\text{under}}$="+f"{params['lambda_under']};    "+\
            r"$\theta_{\text{queue}}$="+f"{params['theta_queue']}", fontsize=12)

    ax.plot(tss, np.cumsum(costs), label="C & K Strategy")
    ax.plot(tss, np.cumsum(costs_bls["BestAsk"]), alpha=0.5, linestyle='--', label='Best Ask')
    ax.plot(tss, np.cumsum(costs_bls["VWAP"]), alpha=0.5, linestyle='--', label='VWAP')
    ax.plot(tss, np.cumsum(costs_bls["TWAP"]), alpha=0.5, linestyle='--', label='TWAP')
    plt.xticks(rotation=15, ha='right')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))

    #ax.set_xlabel("Time", fontsize=12)
    ax.set_xlabel("Number of snapshots", fontsize=12)
    ax.set_ylabel("Cummulative cost ($)", fontsize=12)

    plt.legend()
    plt.tight_layout()
    fig.savefig(out)


def print_result(params: dict, costs: float, costs_bls: dict, avg_price: float, avg_price_bls: dict):
    results = {
            "Best params": {
                "lambda_over": params['lambda_over'],
                "lambda_uncer": params['lambda_under'],
                "theta_queue": params['theta_queue']
                },
            "Results": {
                "CK Strategy": {
                    "total cash spent": sum(costs),
                    "average fill price": avg_price
                    },
                "Baselines": {
                    "Best Ask": {
                        "total cash spent": sum(costs_bls['BestAsk']),
                        "average fill price": avg_price_bls['BestAsk'],
                        "saving": round((avg_price_bls['BestAsk'] - avg_price)*10_000)
                        },
                    "TWAP": {
                        "total cash spent": sum(costs_bls['TWAP']),
                        "average fill price": avg_price_bls['TWAP'],
                        "saving": round((avg_price_bls['TWAP'] - avg_price)*10_000)
                        },
                    "VWAP": {
                        "total cash spent": sum(costs_bls['VWAP']),
                        "average fill price": avg_price_bls['VWAP'],
                        "saving": round((avg_price_bls['VWAP'] - avg_price)*10_000)
                        }
                    },
                "Savins": {
                    "Best Ask": round((avg_price_bls['BestAsk'] - avg_price)*10_000),
                    "TWAP": round((avg_price_bls['TWAP'] - avg_price)*10_000),
                    "VWAP": round((avg_price_bls['VWAP'] - avg_price)*10_000)
                    }
                }
            }
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()

    # Initialize the tunable paramters. If opt is on, then they will be updated
    lambda_over = args.lambda_over
    lambda_under = args.lambda_under
    theta_queue = args.theta_queue

    # Load and clean L1 data
    snapshots = load_and_clean_l1_data(args.file)

    opt_config_path = args.optimize_config
    logger.info(f"Optimize: {opt_config_path}")


    # Optimization
    if opt_config_path:
        with open(opt_config_path, "r") as file:
            opt_config = yaml.safe_load(file)

        algo = opt_config['optimizer']['algorithm']
        num_iterations = opt_config['optimizer']['num_iterations']
        lambda_over_range = opt_config['optimizer']['lambda_over_range']
        lambda_under_range = opt_config['optimizer']['lambda_under_range']
        theta_queue_range = opt_config['optimizer']['theta_queue_range']

        logger.info(f"Optimizing with {algo} ...")
        for k, param in opt_config['optimizer'].items():
            logger.info(f"\t\t{k}: {param}")
        lo_rng = np.linspace(*lambda_over_range, num_iterations)
        lu_rng = np.linspace(*lambda_under_range, num_iterations)
        th_rng = np.linspace(*theta_queue_range, num_iterations)
        min_costs = float("inf")
        for lo in lo_rng:
            for lu in lu_rng:
                for th in th_rng:
                    if lo==0 and lu==0 and th==0: continue
                    costs, _,  sucess = backtest(snapshots, args.order_size, lo, lu, th, args.fee, args.rebate, False, True)
                    if not sucess: continue
                    total_cost = np.sum(costs[0])
                    if min_costs > total_cost:
                        min_costs = total_cost
                        lambda_over = lo
                        lambda_under = lu
                        theta_queue = th

        logger.info(f"Finished optimization.")


    params = {"lambda_over": lambda_over, "lambda_under": lambda_under, "theta_queue": theta_queue}

    logger.info(f"Starting backtest with params: "+\
            f"lambda_over={params['lambda_over']}; "+\
            f"lambda_under={params['lambda_under']}; "+\
            f"theta_queue={params['theta_queue']}")

    # Run the backtest with the parsed arguments
    (costs, costs_bls), (avg_price, avg_price_bls), sucess = backtest(snapshots, args.order_size, lambda_over, lambda_under, theta_queue,
                                                                        args.fee, args.rebate, True, args.early_stop)


    logger.info(f"Save plot to {args.plot_path}")
    plot_cost(snapshots, costs, costs_bls, params, args.plot_path)

    logger.info(f"Printing results... \n\n")
    print_result(params, costs, costs_bls, avg_price, avg_price_bls)


