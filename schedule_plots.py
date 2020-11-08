import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

EPOCHS = 100


# %% PLOT LOGS
def read_file_logs(file):
    with open(file) as f:
        col_names = f.readline().strip("\n").split(",")
    data = np.genfromtxt(file, delimiter=",", skip_header=1)
    logs = {col_name: values for col_name, values in zip(col_names, np.transpose(data))}
    return logs


log_files = [
    "outputs/vitos_drop01/log.csv",
    "outputs/vitos_b2-98_drop01/log.csv",
    "outputs/vitos_b2-98_drop01_b1024/log.csv",
    "outputs/vitos_b2-98_drop01_b1024_sch/log.csv",
    "outputs/vitos_b2-98_drop01_b1024_e200/log.csv"
    # "outputs/vitos_b2-98_drop01_b1024_e200_seed/log.csv"
]
metrics = ["val_loss", "val_accuracy"]
fig, axes = plt.subplots(len(metrics), 1)
for i, metric in enumerate(metrics):
    for j, file in enumerate(log_files):
        logs = read_file_logs(file)
        axes[i].plot(logs[metric], label="file_{}: {}".format(j, metric))
        axes[i].legend()
plt.show()

# %% PLOT SCHEDULES
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

]
for i in range(len(schedules)):
    sch = schedules[i]
    lr_vals = [sch(e) for e in range(EPOCHS)]
    plt.plot(lr_vals, label=sch.name)

plt.legend()
plt.show()
