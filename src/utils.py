import torch

def cycle(iterable):
    """
    An iterator that cycles through the dataset indefinitely.
    This replaces the need for 'for epoch in range(epochs):'
    """
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            # When we finish the dataset, we restart it.
            # In Distributed settings, we would ideally set the epoch here 
            # for the sampler to reshuffle differently next time.
            iterator = iter(iterable)