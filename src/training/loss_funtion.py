"""
src/training/loss_functions.py
==============================
Custom loss functions for the multi-task LSTM training objective.
"""

from __future__ import annotations

try:
    import tensorflow as tf
    from keras._tf_keras import keras

    class FocalBinaryCrossentropy(keras.losses.Loss):
        """
        Focal loss for class-imbalanced binary targets (dropout risk,
        mastery) — down-weights easy examples so the model focuses on
        hard / minority cases.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

        Parameters
        ----------
        gamma : focusing parameter (0 = standard BCE)
        alpha : weight for positive class
        """

        def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs):
            super().__init__(**kwargs)
            self.gamma = gamma
            self.alpha = alpha

        def call(self, y_true, y_pred):
            y_true  = tf.cast(y_true, tf.float32)
            y_pred  = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            bce     = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
            p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            focal_w = (1 - p_t) ** self.gamma
            alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
            return tf.reduce_mean(alpha_t * focal_w * bce)

        def get_config(self):
            return {"gamma": self.gamma, "alpha": self.alpha, **super().get_config()}


    class WeightedMSE(keras.losses.Loss):
        """
        MSE with higher penalty for under-predicting low-performing learners
        (important for early dropout detection).
        """

        def __init__(self, low_score_weight: float = 2.0, threshold: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self.low_score_weight = low_score_weight
            self.threshold = threshold

        def call(self, y_true, y_pred):
            y_true  = tf.cast(y_true, tf.float32)
            sq_err  = tf.square(y_true - y_pred)
            weights = tf.where(y_true < self.threshold,
                               self.low_score_weight, 1.0)
            return tf.reduce_mean(weights * sq_err)

        def get_config(self):
            return {
                "low_score_weight": self.low_score_weight,
                "threshold": self.threshold,
                **super().get_config(),
            }

except ImportError:
    pass   # TF not available — stubs are fine for import resolution