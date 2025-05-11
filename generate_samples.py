#!/usr/bin/env python3
import csv
import os
import random

# ------------------ CONFIGURATION ------------------
MIN_DIGITS   = 1           # minimum digits per operand
MAX_DIGITS   = 3           # maximum digits per operand
TOTAL        = 1000000     # total samples in train+test+val
TRAIN_SIZE   = int(0.8*TOTAL)   # examples in train split
TEST_SIZE    = int(0.1*TOTAL)   # examples in test split
VAL_SIZE     = int(0.1*TOTAL)   # examples in val split
GEN_SIZE     = int(0.1*TOTAL)   # examples in generatlization set
EXTRA_MIN    = 0           # add at least this many digits for generalization
EXTRA_MAX    = 3           # add up to this many digits for generalization
EDGE_FRAC    = 0.8         # fraction of each split reserved for edge cases
SEED         = 42          # set to None for non‑deterministic
OUTPUT_DIR   = "data"      # where to write CSVs
# ---------------------------------------------------------

def generate_example(min_d: int, max_d: int):
    """Random example, allowing negative subtraction results."""
    op = random.choice(["+", "-"])
    low, high = 10**(min_d - 1), 10**max_d - 1
    n1 = random.randint(low, high)
    n2 = random.randint(low, high)
    target = n1 + n2 if op == "+" else n1 - n2
    return str(str(n1)+op+str(n2)), str(target)

def generate_carry_example(min_d: int, max_d: int):
    """Addition example that forces at least one digit‑carry."""
    length = random.randint(min_d, max_d)
    d1 = random.randint(5, 9)
    d2 = random.randint(5, 9)
    def prefix():
        return "".join(str(random.randint(0,9)) for _ in range(length-1))
    n1 = int(prefix() + str(d1))
    n2 = int(prefix() + str(d2))
    target = n1 + n2
    return str(n1)+"+"+str(n2), str(target)

def generate_borrow_example(min_d: int, max_d: int):
    """Subtraction example that forces at least one digit‑borrow, allowing negative results."""
    length = random.randint(min_d, max_d)
    d1 = random.randint(0, 4)
    d2 = random.randint(d1+1, 9)
    def prefix():
        return "".join(str(random.randint(0,9)) for _ in range(length-1))
    n1 = int(prefix() + str(d1))
    n2 = int(prefix() + str(d2))
    target = n1 - n2
    return str(n1)+"-"+str(n2), str(target)

def write_split(path: str, n_examples: int,
                min_d: int, max_d: int,
                edge_frac: float = 0.0,
                seed: int = None):
    """Generate and write `n_examples` to CSV at `path`."""
    if seed is not None:
        random.seed(seed)
    n_edge = int(n_examples * edge_frac)
    n_each = n_edge // 2

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "target"])

        # edge cases
        for _ in range(n_each):
            writer.writerow(generate_carry_example(min_d, max_d))
        for _ in range(n_each):
            writer.writerow(generate_borrow_example(min_d, max_d))

        # random cases
        for _ in range(n_examples - 2*n_each):
            writer.writerow(generate_example(min_d, max_d))

if __name__ == "__main__":
    if SEED is not None:
        random.seed(SEED)

    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(OUTPUT_DIR, "train.csv")
    write_split(
        out_path,
        TRAIN_SIZE,
        MIN_DIGITS, MAX_DIGITS,
        edge_frac=EDGE_FRAC,
        seed=SEED
    )
    out_path = os.path.join(OUTPUT_DIR, "val.csv")
    write_split(
        out_path,
        VAL_SIZE,
        MIN_DIGITS, MAX_DIGITS,
        edge_frac=EDGE_FRAC,
        seed=SEED
    )
    out_path = os.path.join(OUTPUT_DIR, "test.csv")
    write_split(
        out_path,
        TEST_SIZE,
        MIN_DIGITS, MAX_DIGITS,
        edge_frac=EDGE_FRAC,
        seed=SEED
    )

    # 2) generalization: bumped digit range
    gen_min = MIN_DIGITS + EXTRA_MIN
    gen_max = MAX_DIGITS + EXTRA_MAX
    gen_path = os.path.join(OUTPUT_DIR, "generalization.csv")
    write_split(
        gen_path,
        GEN_SIZE,
        gen_min, gen_max,
        edge_frac=EDGE_FRAC,
        seed=SEED
    )

    print(f"✓ Datasets generated in `{OUTPUT_DIR}`:")
    print("  • train.csv")
    print("  • val.csv")
    print("  • test.csv")
    print("  • generalization.csv")
