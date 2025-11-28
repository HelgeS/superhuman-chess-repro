import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import glob

all_files = glob.glob("results/results_*.csv")

dfs = []

for file in all_files:
    print(file)
    df = pd.read_csv(file)

    if "forced_move_positions" in file:
        df_type = "forced_moves"
    elif "mirroring" in file:
        df_type = "mirroring"
        continue
    # elif "recommended" in file:
    #     df_type = "recommended_moves"
    # elif "transformations" in file:
    #     df_type = "transformations"
    # else:
    #     df_type = "unknown"

    df["check"] = df_type

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print(df.head())


table = Table(title="Chess Consistency Checks")

headline = [
    "Consistency check",
    "Samples",
    ">0.05",
    ">0.1",
    ">0.25",
    ">0.5",
    ">0.75",
    ">1.0",
]

for c in headline:
    table.add_column(c, justify="right", style="cyan", no_wrap=True)

thresholds = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]


for check in df["check"].unique():
    df_sub = df[df["check"] == check]

    row = [check, str(len(df_sub))]

    for t in thresholds:
        percentage = (df_sub["differenceconv"] > t).mean()
        row.append(f"{percentage:.2%}")

    table.add_row(*row)

console = Console()
console.print(table)
