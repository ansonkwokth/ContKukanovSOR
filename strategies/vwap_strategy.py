def vwap_strategy(order_size, venues):
    """
    """
    total_size = sum(venue.ask_size for venue in venues)

    if total_size == 0:
        return [0] * len(venues)

    # Step 3: Calculate the weights for each venue based on their ask size
    allocation = [venue.ask_size for venue in venues]


    return allocation
