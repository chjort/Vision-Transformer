import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, learning_rate, warmup_steps, name=None):
        super().__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.name = name

        if isinstance(learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule):
            self.lr_is_schedule = True
            self.warmup_rates = tf.linspace(0.0, learning_rate.initial_learning_rate, warmup_steps)
        else:
            self.lr_is_schedule = False
            self.warmup_rates = tf.linspace(0.0, learning_rate, warmup_steps)

    @tf.function
    def __call__(self, step):
        if step < self.warmup_steps - 1:
            step = tf.cast(step, tf.int32)
            return self.warmup_rates[step]
        else:
            if self.lr_is_schedule:
                return self.learning_rate(step - self.warmup_steps + 1)
            else:
                return self.learning_rate

    def get_config(self):
        return {
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "name": self.name
        }


tf.keras.utils.get_custom_objects().update({
    "LinearWarmup": LinearWarmup,
})
