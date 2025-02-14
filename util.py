def ordered_sample_indices(lst, n):
    """
    Sample n elements from a list in order, return indices
    """

    if n == 0:
        return []
    if n == 1:
        return [0]
    if n >= len(lst):
        return list(range(len(lst)))
    # Calculate the step size based on list length and number of samples
    step = (len(lst) - 1) / (n - 1)
    # Use the calculated step to select indices
    indices = [round(i * step) for i in range(n)]
    return indices


def ordered_sample(lst, n):
    """
    Sample n elements from a list in order.
    """
    indices = ordered_sample_indices(lst, n)
    return [lst[i] for i in indices]
