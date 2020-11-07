import matplotlib.pyplot as plt
import tensorflow as tf

EPOCHS = 100

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
