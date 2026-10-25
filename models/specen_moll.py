import numpy as np
import tensorflow as tf
from tensorflow.keras import Model,layers,losses

class SpecEnMoll(Model):
    def __init__(self,input_shape,output_shape,optimizer=None):
        super().__init__()
        lamb=0.0
        self.ffn = tf.keras.Sequential([
            layers.Dense(input_shape, activation='relu',kernel_regularizer=tf.keras.regularizers.l1(l1=lamb)),
            layers.Dense(input_shape//2, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(l1=lamb)),
            layers.Dense(output_shape,activation='softmax',kernel_regularizer=tf.keras.regularizers.l1(l1=lamb)),
        ])
        self.optimizer=optimizer

    def call(self,x):
        return self.ffn(x)

    def loss(self, y_pre, prior_pre,y_true):
        p=tf.reduce_sum(y_pre*prior_pre,axis=-1)
        return losses.binary_crossentropy(y_true,p)

    def train_step(self,x,y,prior_pre):
        with tf.GradientTape() as tape:
            y_pred = self.call(x)
            l=self.loss(y_pred,prior_pre,y)
            grads = tape.gradient(l,self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.trainable_variables))
        return l

    def predict1(self,x,prior_proba,batch_size):
        y_pred=tf.cast(
            tf.reduce_sum(self.predict(x, batch_size=batch_size) * prior_proba, axis=-1) >= 0.5,
            dtype=tf.int32)
        return y_pred
