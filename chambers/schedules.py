import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, schedule, warmup_steps, name=None):
        super().__init__()
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.name = name
        self.warmup_rates = tf.linspace(0.0, schedule.initial_learning_rate, warmup_steps)

    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps - 1:
            step = tf.cast(step, tf.int32)
            return self.warmup_rates[step]
        else:
            return self.schedule(step - self.warmup_steps + 1)

    def get_config(self):
        return {
            "schedule": self.schedule,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }
