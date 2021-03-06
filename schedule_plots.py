# %% PLOT LOGS
import matplotlib.pyplot as plt
import numpy as np


def read_file_logs(file):
    with open(file) as f:
        col_names = f.readline().strip("\n").split(",")
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    logs = {col_name: values for col_name, values in zip(col_names, np.transpose(data))}
    return logs


log_files = [
    # "outputs/vitos_drop01/log.csv",
    # "outputs/vitos_b2-98_drop01/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024_sch/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024_e200/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024_e200_seed42/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024_e200_seed41/log.csv",
    # "outputs/vitos_b2-98_drop01_b1024_e200_1/log.csv",
    "outputs/vitos_b2-98_drop01_b256_e200_p7/log.csv",
    "outputs/vitos_b2-98_drop01_b256_e200_p7_1/log.csv",
    "outputs/vitos_b2-98_drop01_b256_e200_p7_2/log.csv",
    "outputs/vitos_sched/log.csv",
    "outputs/vitos21_p7_e200_bz256/log.csv",
    "outputs/vitos2_p7_e220_bz256_sch/log.csv",
    # "outputs/vitos_sin/log.csv",
    # "outputs/vitos_sin2d/log.csv",
    # "outputs/v2vitos_p4/log.csv"  # is underfitting
]
metrics = ["loss", "accuracy"]
fig, axes = plt.subplots(len(metrics), 2)
for i, metric in enumerate(metrics):
    for j, file in enumerate(log_files):
        logs = read_file_logs(file)
        axes[i][0].plot(logs[metric], label="file_{}: {}".format(j, metric))
        axes[i][0].legend()

        val_metric = "val_" + metric
        axes[i][1].plot(logs[val_metric], label="file_{}: {}".format(j, val_metric))
        axes[i][1].legend()

        if i == 0:
            axes[i][0].set_title("Train")
            axes[i][1].set_title("Val")
plt.show()

# %% PLOT SCHEDULES
import tensorflow as tf
import matplotlib.pyplot as plt
from chambers.schedules import LinearWarmup

EPOCHS = 100
LR = 0.00006
schedules = [
    tf.keras.optimizers.schedules.ExponentialDecay(LR,
                                                   decay_steps=66,
                                                   decay_rate=0.1,
                                                   staircase=False,
                                                   name="exp"),
    tf.keras.optimizers.schedules.ExponentialDecay(LR,
                                                   decay_steps=66,
                                                   decay_rate=0.1,
                                                   staircase=True,
                                                   name="exp_stair"),
    tf.keras.optimizers.schedules.InverseTimeDecay(LR,
                                                   decay_steps=66,
                                                   decay_rate=0.1,
                                                   staircase=False,
                                                   name="inverse"),
    tf.keras.optimizers.schedules.InverseTimeDecay(LR,
                                                   decay_steps=66,
                                                   decay_rate=0.1,
                                                   staircase=True,
                                                   name="inverse_stair"),
    tf.keras.optimizers.schedules.PolynomialDecay(LR,
                                                  decay_steps=66,
                                                  end_learning_rate=0.000006,
                                                  cycle=False,
                                                  name="poly"),
    tf.keras.optimizers.schedules.PolynomialDecay(LR,
                                                  decay_steps=66,
                                                  end_learning_rate=0.000006,
                                                  cycle=True,
                                                  name="poly_cycle"),
    LR
]
for i in range(len(schedules)):
    sch = LinearWarmup(schedules[i], warmup_steps=10)
    lr_vals = [sch(tf.constant(e)) for e in range(EPOCHS)]
    if sch.lr_is_schedule:
        plt.plot(lr_vals, label=sch.learning_rate.name)
    else:
        plt.plot(lr_vals, label=sch.learning_rate)

plt.legend()
plt.show()