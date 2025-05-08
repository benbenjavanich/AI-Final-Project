import matplotlib.pyplot as plt
import pandas as pd

def plot_response_times(rt, title):
    plt.figure()
    plt.hist(rt, bins=20)
    plt.title(title)
    plt.xlabel("Response Time (min)")
    plt.ylabel("Count")
    # instead of plt.show(), save and close:
    fname = f"{title}_response_times.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved response‐time histogram to {fname}")

def plot_workload(assignments, title):
    df = pd.DataFrame(assignments, columns=["patient", "cna", "time"])
    counts = df["cna"].value_counts().sort_index()
    plt.figure()
    counts.plot.bar()
    plt.title(f"Workload per CNA — {title}")
    plt.xlabel("CNA ID")
    plt.ylabel("Number of Visits")
    fname = f"{title}_workload.png"
    plt.savefig(fname)
    plt.close()
    print(f"Saved workload bar chart to {fname}")
