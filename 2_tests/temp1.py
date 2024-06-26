def divide_list(lst, n):
    """
    Divide a list into n roughly equal parts.

    Parameters:
    lst (list): The list to be divided.
    n (int): The number of parts to divide the list into.

    Returns:
    list of lists: A list containing n sublists with the divided elements.
    """
    # Determine the size of each part
    avg = len(lst) / float(n)
    out = []
    last = 0.0

    while last < len(lst):
        out.append(lst[int(last):int(last + avg)])
        last += avg

    return out


# Example usage
my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
num_parts = 20
divided_list = divide_list(my_list, num_parts)
print(divided_list)