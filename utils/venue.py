import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from utils.logger import get_logger

logger = get_logger('venue_class', logging.INFO)

class Venue:
    """
    A class representing a market venue, which provides market data for order allocation.

    Attributes:
        ask (float): The best ask price at this venue.
        ask_size (int): The size of the order available at the best ask price.
        fee (float): The fee associated with placing an order at this venue.
        rebate (float): The rebate provided for liquidity provision at this venue.

    Methods:
        __repr__(): Returns a string representation of the Venue instance for easy inspection.
        __str__(): Returns a more user-friendly string representation of the Venue instance.
    """

    def __init__(self, ask: float, ask_size: int, fee: float, rebate: float):
        self.ask = ask
        self.ask_size = ask_size
        self.fee = fee
        self.rebate = rebate

    def __repr__(self):
        return f"Venue(ask={self.ask}, ask_size={self.ask_size}, fee={self.fee}, rebate={self.rebate})"

    def __str__(self):
        return f"Ask={self.ask}; Ask size={self.ask_size}; Fee={self.fee}; Rebate={self.rebate}"


# Quick test for Venue class
if __name__ == "__main__":
    venue = Venue(ask=100.5, ask_size=200, fee=0.001, rebate=0.0005)

    # Test the __str__ method (user-facing output)
    logger.info("User-facing output:")
    logger.info(venue)  # This calls the __str__ method

    # Test the __repr__ method (developer-facing output)
    logger.info("Developer-facing output:")
    logger.info(repr(venue))  # This calls the __repr__ method

