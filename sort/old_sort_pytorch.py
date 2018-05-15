'''

    THIS IS NOT FULLY FUNCTIONAL, please
    see tf_sort.py for code used for the final project implemented in Tensorflow.

    Initial implementation in Pytorch 0.4.0.

'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def generate_dataset(input_sz, num_samples, min_val=0, max_val=100):
    raw =  np.random.randint(min_val, max_val, size=(num_samples, input_sz))
    return raw.astype(np.float32), np.sort(raw).astype(np.float32)


class PlasticFeedforwardNN(nn.Module):

    def __init__(self, input_sz):
        super(PlasticFeedforwardNN, self).__init__()

        self.input_sz = input_sz
        self.hidden_sz1 = 40
        self.hidden_sz2 = 40
        # self.hidden_sz3 = 3

        self.ff1 = nn.Linear(input_sz, 40)
        self.ff2 = nn.Linear(self.hidden_sz1, self.hidden_sz2)

        self.eta = 0.01
        # self.eta1 = torch.tensor(0.01 * torch.ones(1), requires_grad=True)
        self.w1 = torch.tensor(0.01 * torch.randn(self.hidden_sz2, self.input_sz), requires_grad = True)
        self.alphas1 = torch.tensor(0.01 * torch.randn(self.hidden_sz2, self.input_sz), requires_grad=True)

        # self.eta2 = torch.tensor(0.01 * torch.ones(1), requires_grad=True)
        # self.w2 = torch.tensor(0.01 * torch.randn(self.hidden_sz1, self.hidden_sz2), requires_grad = True)
        # self.alphas2 = torch.tensor(0.01 * torch.randn(self.hidden_sz1, self.hidden_sz2), requires_grad=True)



    def forward(self, x, hebb):

        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x))

        # First Plastic Layer
        # print(hebb2)
        # print('a', torch.mm(x, self.w1))
        # print('alpahs term', torch.mm(x, self.alphas1*hebb1))
        out1 = torch.mm(x, self.w1) + torch.mm(x, self.alphas1*hebb)
        # print(out1)
        hebb = (1 - self.eta)*hebb + self.eta*torch.mm(torch.t(x), out1)

        return out1, hebb


    def initialize_hebb1(self):
        return torch.zeros(self.hidden_sz2, self.input_sz)

    # def initialize_hebb2(self):
    #     return torch.zeros(self.hidden_sz1, self.hidden_sz2)


if __name__ == '__main__':
    sequence_len = 3
    num_train, num_test = 1000000, 150

    train_data, train_labels = generate_dataset(sequence_len, num_train)
    test_data, test_labels = generate_dataset(sequence_len, num_test)

    criterion = nn.MSELoss()

    net = PlasticFeedforwardNN(input_sz=sequence_len)
    params = list(net.parameters()) + [net.w1, net.alphas1]
    optimizer = optim.SGD(params, lr=0.00001)


    # Train Loop

    batch_sz = 1
    hebb = net.initialize_hebb1()
    for i in range(train_data.shape[0]//batch_sz):
        if i % 1000 == 0:
            print('Iteration: {}'.format(i))
            # hebb1 = net.initialize_hebb1()
        optimizer.zero_grad()   # zero the gradient buffers
        max_val = train_data[i, :].max()
        output, hebb = net(torch.from_numpy(train_data[i*batch_sz:(i+1)*batch_sz, :]/max_val), hebb)
        output = np.asscalar(max_val)*output
        loss = criterion(output, torch.from_numpy(train_labels[i*batch_sz:(i+1)*batch_sz, :]))
        loss.backward()
        optimizer.step()    # Does the update


    # Compute Testing Accuracy:

    # Metric 1 MSE:
    test_guesses = np.zeros_like(test_labels)
    for i in range(test_data.shape[0]):
        hebb = net.initialize_hebb1()
        out, hebb = net(torch.from_numpy(test_data[i: i + 1, :]), hebb)
        test_guesses[i, :] = out.detach().numpy()
    # test_guesses = net(torch.from_numpy(test_data)).detach().numpy()
    mse = np.sum(np.square(test_guesses - test_labels))/test_data.shape[0]
    print('MSE:', mse)

    # Metric 2 (Rounded) 0-1 loss:
    rounded = np.rint(test_guesses)
    print('Mistakes', (rounded != test_labels).sum())
    print(test_labels.size)
    print('0-1 loss: ', (rounded != test_labels).sum()/test_labels.size)
