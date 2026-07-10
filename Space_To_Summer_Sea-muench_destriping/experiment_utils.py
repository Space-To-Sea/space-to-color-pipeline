```python
"""
experiment_utils.py

a few helper functions for running the muench parameter tests.
"""

import importlib
import random


def clean_float(x):
    """
    make a float safe to use in folder/file names.

    example:
        2.40 -> 2p40
    """
    return f"{float(x):.2f}".replace(".", "p")


def latin_hypercube_samples(n, low, high, is_int=False):
    """
    generate samples across the full parameter range using
    latin hypercube sampling.

    basically splits the range into sections, picks one random
    value from each, then mixes up the order. gives us better
    coverage than just picking everything completely at random.

    args:
        n: how many values to generate
        low: bottom of the range
        high: top of the range
        is_int: round values to integers if needed

    returns:
        list of sampled values
    """

    # need at least one sample
    if n <= 0:
        raise ValueError("n must be greater than 0")

    # range has to go in the right direction
    if high < low:
        raise ValueError("high must be >= low")

    # nothing to sample if both ends are the same
    if high == low:
        return [int(low) if is_int else float(low)] * n

    # split the full range into equal sections
    step = (high - low) / n
    values = []

    for i in range(n):
        # get the bounds for this section
        bin_low = low + i * step
        bin_high = low + (i + 1) * step

        # pick one random value somewhere inside it
        value = random.uniform(bin_low, bin_high)

        if is_int:
            # round it and make sure it stays inside the range
            value = int(round(value))
            value = max(int(low), min(int(high), value))

        values.append(value)

    # mix them up so the tests don't just run low to high
    random.shuffle(values)

    return values


def safe_import_process_rgba(module_name):
    """
    load process_rgba() from the pipeline file.

    example:
        module_name='modified_muench'
        loads it from modified_muench.py
    """

    try:
        # import whichever pipeline module was passed in
        module = importlib.import_module(module_name)

    except ImportError as exc:
        raise ImportError(
            f"Could not import module '{module_name}'. "
            f"Make sure {module_name}.py is in the same folder as sweep.py, "
            f"or run this from the repo root."
        ) from exc

    # quick check that the pipeline has the function we need
    if not hasattr(module, "process_rgba"):
        raise AttributeError(
            f"Module '{module_name}' does not contain process_rgba(). "
            f"Check your pipeline file name or function name."
        )

    # send the pipeline function back to sweep.py
    return module.process_rgba
```
