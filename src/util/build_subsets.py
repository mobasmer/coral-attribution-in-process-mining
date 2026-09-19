import itertools
import random

def powerset(iterable, sample_size: int = 0):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    if sample_size == 0:
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))
    else:
        s = list(iterable)
        i = 0
        subsets = set()
        while sample_size > i:
            subsets.add(random_subset_uniform(s))
            i += 1
        return subsets

def random_combination(players, sample_size):
    n = len(players)
    indices = sorted(random.sample(range(n), sample_size))
    return tuple(players[i] for i in indices)

def random_subset_uniform(players):
    "Uniform sample from the powerset (each element included independently, p=0.5)"
    n = len(players)
    r = sum(random.random() < 0.5 for _ in range(n))  # ~ Binomial(n, 0.5)
    return random_combination(players, r)