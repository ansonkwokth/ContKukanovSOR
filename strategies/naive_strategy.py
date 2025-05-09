def naive_strategy(order_size, venues):
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
    venues_sorted = sorted(venues, key=lambda v: v.ask)

    allocation = []
    remaining_order = order_size
    total_cost = 0.0

    # Fill the order with the best ask first, then move to the next best
    for venue in venues_sorted:
        if remaining_order <= 0:
            break

        # Take as many shares as possible from this venue
        qty_to_buy = min(remaining_order, venue.ask_size)
        allocation.append(qty_to_buy)

        # Update total cost (ask price * quantity)
        total_cost += qty_to_buy * venue.ask

        # Update remaining order
        remaining_order -= qty_to_buy

    # Fill the remaining allocation with 0 if the order was filled partially
    allocation += [0] * (len(venues) - len(allocation))

    return allocation, total_cost

