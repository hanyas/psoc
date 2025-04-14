import numpy as np

from matplotlib import pyplot as plt


def beautify(ax):
    ax.set_frame_on(True)
    ax.minorticks_on()

    ax.grid(True)
    ax.grid(linestyle=":")

    ax.tick_params(
        which="both",
        direction="in",
        bottom=True,
        labelbottom=True,
        top=True,
        labeltop=False,
        right=True,
        labelright=False,
        left=True,
        labelleft=True,
    )

    ax.tick_params(which="major", length=6)
    ax.tick_params(which="minor", length=3)

    # ax.autoscale(tight=True)
    # ax.set_aspect('equal')

    if ax.get_legend():
        ax.legend(loc="best")

    return ax


env_path = ""
env_name = "double_pendulum"

ppo_reward = (-1.0 * np.load(str(env_path) + "ppo_" + str(env_name) + ".npy")[:, 1:].T)[:, ::3]
trpo_reward = (-1.0 * np.load(str(env_path) + "trpo_" + str(env_name) + ".npy")[:, 1:].T)[:, ::3]
smc_reward = -1.0 * np.load(str(env_path) + "smc_" + str(env_name) + ".npy")[:, ::15]
csmc_reward = -1.0 * np.load(str(env_path) + "csmc_" + str(env_name) + ".npy")[:, ::15]

ppo_mean = np.mean(ppo_reward, axis=0)
trpo_mean = np.mean(trpo_reward, axis=0)
smc_mean = np.mean(smc_reward, axis=0)
csmc_mean = np.mean(csmc_reward, axis=0)

ppo_std = np.std(ppo_reward, axis=0)
trpo_std = np.std(trpo_reward, axis=0)
smc_std = np.std(smc_reward, axis=0)
csmc_std = np.std(csmc_reward, axis=0)

iters = np.linspace(0, 60, 21, dtype=int)

ax = plt.gca()
ax.errorbar(iters, ppo_mean, ppo_std, fmt="-o")
ax.errorbar(iters, trpo_mean, trpo_std, fmt="-o")
ax.errorbar(iters, smc_mean, smc_std, fmt="-o")
ax.errorbar(iters, csmc_mean, csmc_std, fmt="-o")
ax = beautify(ax)
# plt.show()

import tikzplotlib
tikzplotlib.save("double_pendulum.tex")
