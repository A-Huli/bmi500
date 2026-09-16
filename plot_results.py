import os
import pstats
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

datasets = ["pbmc3k", "pbmc6k", "pbmc10k"]

df = pd.concat([pd.read_csv(f"data/{d}_timings.csv").assign(dataset=d) for d in datasets])
sections = df["section"].unique()
table = df.pivot(index="dataset", columns="section", values="seconds").loc[datasets, sections]
cells = df.groupby("dataset")["n_cells"].first().loc[datasets]

print("Instrumented profile (seconds):")
print(table.T.round(3).to_string())
print("TOTAL:", table.sum(axis=1).round(3).to_dict())

labels = [f"{d}\n({n} cells)" for d, n in cells.items()]
ax = table.set_axis(labels).plot(kind="bar", stacked=True, colormap="tab20", figsize=(9, 6), rot=0)
ax.set_xlabel("Dataset")
ax.set_ylabel("Run time (s)")
ax.set_title("Instrumented profile: run time per code section")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig("instrumented_profile.png", dpi=150)
plt.close()

for d in datasets:
    stats = pstats.Stats(f"data/{d}.prof").stats
    top = sorted(stats.items(), key=lambda kv: kv[1][3], reverse=True)[:15]
    names = [f"{os.path.basename(f)}:{line}({func})" for (f, line, func), _ in top]
    cumtime = [v[3] for _, v in top]
    plt.figure(figsize=(9, 6))
    plt.barh(names[::-1], cumtime[::-1])
    plt.xlabel("Cumulative time (s)")
    plt.title(f"cProfile of rank_genes_groups on {d}")
    plt.tight_layout()
    plt.savefig(f"cprofile_{d}.png", dpi=150)
    plt.close()

print("\nExpected run time at 20,000 cells:")
for s in list(sections) + ["TOTAL"]:
    t = table.sum(axis=1) if s == "TOTAL" else table[s]
    b, log_a = np.polyfit(np.log(cells), np.log(t), 1)
    print(f"{s:<22} b = {b:5.2f}   20K: {np.exp(log_a) * 20000 ** b:8.2f} s")