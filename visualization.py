import matplotlib.pyplot as plt
import pandas as pd

def plot_pnl_EI(pnl: pd.Series, EI: pd.Series, filename):
    fig, ax1 = plt.subplots(figsize=(8, 4))

    # left y -> pnl
    ax1.plot(pnl.index, pnl, color='tab:blue', label='pnl')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("PnL", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # right y -> EI
    ax2 = ax1.twinx()
    ax2.plot(EI.index, EI, color='tab:orange', label='EI')
    ax2.set_ylabel("EI", color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(-0.05, 1.05)

    plt.title(filename)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.show()
