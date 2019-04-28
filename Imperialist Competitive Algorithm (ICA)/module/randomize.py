from numpy import random
from random import shuffle
from typing import Union, Tuple


def permutation_int(size: Union[int, Tuple[int]]):
    """

    :param size: int
    :return: a list of permutation
    """

    # make a list of random number and get the index of sorted list as a permutation:
    number_array = random.random(size)
    number_array = number_array.argsort()

    return number_array.tolist()


def non_repeat_randint(low, high, size):
    """

    :param low: lower bound: int
    :param high: upper bound: int
    :param size: int
    :return: a randomly list of non_repeated integer numbers
    """
    assert (high - low >= size)

    range_list = list(range(low, high))
    shuffle(range_list)  # inplace function

    '''
    # version 2:
    range_list = list(range(low, high))
    # take one number in range_list randomly and add it to non_repeated_list
    non_repeated_list = [range_list.pop(random.randint(0, len(range_list))) for _ in range(size)]

    # version 1:
    number_array = np.randint(low, high, size)
    number_array = sorted(number_array)

    repeated_idx = []  # to keep index of repeated numbers.
    not_included = []
    # find repeated numbers:
    for i in range(size):
        if number_array[i + 1] - number_array[i] > 1:  # if <there is gap between sorted numbers>:
            # add those numbers that are not in the number_array to not_included:
            start = number_array[i] + 1
            stop = number_array[i + 1]
            not_included.extend(list(range(start, stop)))
        elif number_array[i] == number_array[i + 1]:  # but if <number is repeated>:
            repeated_idx.append(i)  # remember it's place
    # now we have some numbers that are repeated in number_array and some numbers that are not in number_array.
    for idx in repeated_idx:
        random.i
        number_array[idx] = not_included[]
    '''
    return range_list[:size]

# import time
#
# tic = time.process_time()
# permutation_int(1000000)
# toc = time.process_time()
# print("permutation_int: " + str((toc - tic)), 'ms')
#
#
# tic = time.process_time()
# non_repeat_randintv2(0, 1000000, 1000000)
# toc = time.process_time()
# print("non_repeat_randintv2: " + str((toc - tic)), 'ms')
