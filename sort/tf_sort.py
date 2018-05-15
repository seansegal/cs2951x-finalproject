'''

    A feedforward Plastic Neural Network implementation. Replicating the network
    architecture described in "Differentiable plasticity: training plastic neural
    networks with backpropagation" by Stanley et al (2017)

    Authors: Sean Segal & Ansel Vahle
    Written For CS2951X (Professor George Konidaris)

    Written in Tensorflow with Python 2.7.

'''
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


def generate_dataset(input_sz, num_samples, ordering='ascending', min_val=0, max_val=100):
    raw = np.random.randint(min_val, max_val, size=(num_samples, input_sz))
    if ordering == 'ascending':
        return raw.astype(np.float32), np.sort(raw).astype(np.float32)
    if ordering == 'descending':
        return raw.astype(np.float32), np.fliplr(np.sort(raw)).astype(np.float32)


class PlasticFFNeuralNet():
    '''
        A feedforward plastic neural network implemented with Tensorflow. Currently,
        only the first layer of the network is plastic (but this can easily be changed).
    '''
    def __init__(self, h1, h2, input_sz=3, net_type='plastic'):
        self.input_sz = input_sz
        self.hidden_sz1 = h1
        self.hidden_sz2 = h2
        self.learning_rate = 1e-4
        self.type = net_type

        self.x_p = tf.placeholder(tf.float32, shape=self.input_sz)
        self.x = tf.expand_dims(self.x_p, 0)
        self.hebb = tf.placeholder(tf.float32, shape=(
        self.input_sz, self.hidden_sz1), name='hebb')
        self.y_p = tf.placeholder(tf.float32, shape=self.input_sz)
        self.y = tf.expand_dims(self.y_p, 0)

        self.eta = 0.5

        # Alphas determine the weighting between the plastic component and non-plastic component.
        self.alphas = tf.get_variable(
            'alphas_{}_{}_{}'.format(net_type, input_sz, h1), shape=(self.input_sz, self.hidden_sz1))

        self.guess = self._inference(self.x, self.hebb)
        self.new_hebb = self._new_hebb(
            self.x, self.x1, self.hebb, rule_type='hebbian')

        self.loss = self._loss(self.guess, self.y)
        self.train_op = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)


        self.saver = tf.train.Saver()

    def _loss(self, guess, true):
        return tf.losses.mean_squared_error(guess, true)

    def _new_hebb(self, inpt, out, curr_hebb, rule_type='hebbian'):
        if self.type == 'plastic':
            if rule_type == 'hebbian':
                return self.eta * tf.matmul(tf.transpose(inpt), out) + (1 - self.eta) * curr_hebb
            if rule_type == 'oja':
                return curr_hebb + self.eta * (tf.matmul(tf.transpose(inpt), out) - out * curr_hebb)
            raise ValueError('Incorrect rule type')
        return self.hebb # Just return the same thing.


    def save(self, sess, ckptfile):
        save_path = self.saver.save(sess, ckptfile)
        print("Model saved in path: %s" % save_path)

    def _inference(self, x, hebb):

        self.x1 = tf.layers.dense(x, self.hidden_sz1, activation=None)
        if self.type == 'plastic':
            self.x1 = tf.nn.tanh(
                self.x1 + tf.matmul(x, self.alphas * self.hebb))
        elif self.type == 'non-plastic':
            self.x1 = tf.nn.tanh(self.x1)
        else:
            raise ValueError('Type must be plastic or non-plastic.')

        self.x2 = tf.layers.dense(
            self.x1, self.hidden_sz2, activation=tf.nn.relu)
        self.x3 = tf.layers.dense(self.x2, self.input_sz, activation=None)
        return self.x3


def is_correct(inpt, outpt):
    arg_mins = np.abs(inpt.reshape((1, -1)) - outpt.reshape(-1, 1)).argmin(axis = 1)
    return np.allclose(arg_mins, np.arange(0, arg_mins.shape[0]))

def train_and_test(sequence_len, p, model, ckptfile=None, num_episodes=5000, episode_len=50):

    # Initialize a Tensorflow Session.
    with tf.Session() as session:

        # Initialize variables
        session.run(tf.global_variables_initializer())

        # Train Loop
        num_correct, count = 0, 0
        for ep in range(num_episodes):
            if random.random() > p:
                train_data, train_labels = generate_dataset(
                    sequence_len, episode_len, ordering='ascending')
            else:
                train_data, train_labels = generate_dataset(
                    sequence_len, episode_len, ordering='descending')

            # Clear hebbian values for each episode.
            hebb = np.zeros(dtype=np.float32, shape=(sequence_len, hidden_sz1))
            for i in range(train_data.shape[0]):
                inpt = train_data[i, :] / (100)
                outpt = train_labels[i, :] / (100)
                _, loss, guess, hebb = session.run([model.train_op, model.loss, model.guess, model.new_hebb], feed_dict={
                                                 model.x_p: inpt, model.y_p: outpt, model.hebb: hebb})


                count += 1
                num_correct += 1 if is_correct(guess, outpt) else 0
                if np.isnan(hebb).any():
                    print('Hebbian values are nan at Iteraion: {}'.format(i))
                    exit()

            if ep % 10 == 0:
                print('Epsiode: {} | Loss : {} | Correct: {} out of {}'.format(ep, loss, num_correct, count))
                num_correct = 0
                count = 0

        # Save the Tensorflow model
        if not (ckptfile is None):
            model.save(session, ckptfile)

        # Test Loop
        num_test_episodes = 50
        total_loss = 0
        num_correct = 0
        counter = 0
        for ep in range(num_test_episodes):
            if random.random() > p:
                test_data, test_labels = generate_dataset(
                    sequence_len, episode_len, ordering='ascending')
            else:
                test_data, test_labels = generate_dataset(
                    sequence_len, episode_len, ordering='descending')

            # Clear hebbian values for each episode.
            losses = []
            hebb = np.zeros(dtype=np.float32, shape=(sequence_len, hidden_sz1))
            for i in range(test_data.shape[0]):
                inpt = test_data[i, :]/100
                outpt = test_labels[i, :]/100
                if i < episode_len / 2:
                    loss, guess, hebb = session.run([model.loss, model.guess, model.new_hebb], feed_dict={
                                                    model.x_p: inpt, model.y_p: outpt, model.hebb: hebb})
                else:
                    loss, guess, hebb = session.run([model.loss, model.guess, model.new_hebb], feed_dict={
                                                    model.x_p: inpt, model.y_p: outpt, model.hebb: hebb})
                    total_loss += loss
                    counter += 1
                    num_correct += 1 if is_correct(guess, outpt) else 0
                    losses.append(loss)

                if np.isnan(hebb).any():
                    print('Hebbian values are NaN at Iteraion: {}'.format(i))
                    exit()

            if ep % 10 == 0:
                print('In: {}'.format(inpt * 100))
                print('Guess: {}'.format(guess * 100))
                print('Out: {}'.format(outpt * 100))
                print('Epsiode: {} | Loss : {}'.format(ep, loss))

        print('Total Loss: {}'.format(total_loss/num_test_episodes))
        print('Number of correct: {} out of {}'.format(num_correct, counter))


if __name__ == '__main__':

    # Probabability that we get the 'descending' task.
    p = 0.5

    i = 0
    for sequence_len in [3]:
        for hidden in [20]:

            # Hidden sizes of the network.
            hidden_sz1, hidden_sz2 = hidden, hidden

            # Plastic Network
            plastic_net = PlasticFFNeuralNet(hidden_sz1, hidden_sz2,
                              input_sz=sequence_len, net_type='plastic')

            # Baseline Network
            basline_net = PlasticFFNeuralNet(hidden_sz1, hidden_sz2,
                              input_sz=sequence_len, net_type='non-plastic')

            for trial in range(3):
                print('SEQUENCE LENGTH: {} || HIDDEN : {} || Trial : {}'.format(sequence_len, hidden, trial))

                print('---------Network Type: {}-----------'.format(plastic_net.type))
                train_and_test(sequence_len, p,  plastic_net, ckptfile='ckpts/{}-{}-{}-plastic.ckpt'.format(sequence_len, hidden, trial))


                print('--------Network Type: {}------------'.format(basline_net.type))
                train_and_test(sequence_len, p,
                 basline_net, ckptfile='ckpts/{}-{}-{}-regular.ckpt'.format(sequence_len, hidden, trial),)
