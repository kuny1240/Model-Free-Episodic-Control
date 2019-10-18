from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Input,Concatenate
import tensorflow as tf
from tensorflow.python.keras import backend as K

HIDDEN1_UNITS = 128
HIDDEN2_UNITS = 256


class ActorNetwork(object):
    def __init__(self, sess, state_size, embedding_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network(state_size, embedding_size)
        # self.action_gradient = tf.placeholder(tf.float32, [None, embedding_size])
        # self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        # grads = zip(self.params_grad, self.weights)
        # self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })


    def create_actor_network(self, state_size, embedding_size):
        print("Now we build the model")
        S = Input(shape=[state_size])
        h0 = Dense(HIDDEN1_UNITS, activation='relu',kernel_initializer="glorot_normal")(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu',kernel_initializer="glorot_normal")(h0)
        h = Dense(embedding_size,activation='relu',kernel_initializer="glorot_normal")(h1)
        model = Model(inputs =S, outputs = h)
        optimizer = tf.keras.optimizers.Adam(lr=self.LEARNING_RATE,decay=1e-6)
        model.compile(optimizer=optimizer,loss="mse")
        return model, model.trainable_weights, S