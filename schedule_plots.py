import matplotlib.pyplot as plt
import numpy as np


# %% PLOT LOGS
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
    "outputs/vitos_b2-98_drop01_b1024_e200/log.csv",
    "outputs/vitos_b2-98_drop01_b1024_e200_seed42/log.csv",
    "outputs/vitos_b2-98_drop01_b1024_e200_seed41/log.csv",
    "outputs/vitos_b2-98_drop01_b1024_e200_1/log.csv",
    "outputs/vitos_b2-98_drop01_b256_e200_p7/log.csv"
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
]
for i in range(len(schedules)):
    sch = schedules[i]
    lr_vals = [sch(e) for e in range(EPOCHS)]
    plt.plot(lr_vals, label=sch.name)

plt.legend()
plt.show()
