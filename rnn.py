import numpy as np

class RNN:
    def __init__(self, file_name='input.txt'):
        self.vocab_size, self.char_to_ix, self.ix_to_char, self.data = self.dataIO(file_name)
        self.hidden_size = 100                      # neurons of hidden layer
        self.seq_length = 25
        self.learning_rate = 1e-1

        # model parameters
        self.W_xh = np.random.randn(self.hidden_size, self.vocab_size)*0.01
        self.W_hh = np.random.randn(self.hidden_size, self.hidden_size)*0.01
        self.W_hy = np.random.randn(self.vocab_size, self.hidden_size)*0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((self.vocab_size, 1))

        self.mW_xh = np.zeros_like(self.W_xh)
        self.mW_hh = np.zeros_like(self.W_hh)
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)

    @staticmethod
    def dataIO(file_name):
        data = open(file_name, 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print(f'data has {data_size} characters, {vocab_size} unique.')
        char_to_ix = { ch:i for i, ch in enumerate(chars) }
        ix_to_char = { i:ch for i, ch in enumerate(chars) }
        return vocab_size, char_to_ix, ix_to_char, data

    def forward(self, inputs, targets, h_prev):
        x, h, y, p = {}, {}, {}, {}
        h[-1] = np.copy(h_prev)
        loss = 0

        for t in range(len(inputs)):
            x[t] = np.zeros((self.vocab_size, 1))                   #encode in 1-of-k representation
            x[t][inputs[t]] = 1
            h[t] = np.tanh(self.W_xh.dot(x[t]) + self.W_hh.dot(h[t-1]) + self.bh)
            y[t] = self.W_hy.dot(h[t]) + self.by
            p[t] = np.exp(y[t]) / np.sum(np.exp(y[t]))
            loss += -np.log(p[t][targets[t], 0])
        
        return [x, h, y, p, loss]

    def backward(self, inputs, targets, state):
        x, h, y, p, loss = state
        dW_xh, dW_hh, dW_hy = np.zeros_like(self.W_xh), np.zeros_like(self.W_hh), np.zeros_like(self.W_hy)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(h[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(p[t])
            dy[targets[t]] -= 1
            dW_hy += dy.dot(h[t].T)
            dby += dy
            dh = self.W_hy.T.dot(dy) + dh_next
            dh_raw = (1 - h[t] * h[t])*dh
            dbh += dh_raw
            dW_xh += dh_raw.dot(x[t].T)
            dW_hh += dh_raw.dot(h[t-1].T)
            dh_next = self.W_hh.T.dot(dh_raw)
        for d_param in [dW_xh, dW_hh, dW_hy, dbh, dby]:
            np.clip(d_param, -5, 5, out=d_param)
        return loss, dW_xh, dW_hh, dW_hy, dbh, dby, h[len(inputs) - 1]

    def updateAdagrad(self, dW_xh, dW_hh, dW_hy, dbh, dby):
        for param, d_param, mem in zip([self.W_xh, self.W_hh, self.W_hy, self.bh, self.by],
                                        [dW_xh, dW_hh, dW_hy, dbh, dby],
                                        [self.mW_xh, self.mW_hh, self.mW_hy, self.mbh, self.mby]):
            mem += d_param * d_param
            param += -self.learning_rate * d_param / np.sqrt(mem + 1e-8)

    def sample(self, h, seed_ix, n):
        '''
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        '''
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = list()
        for t in range(n):
            h = np.tanh(self.W_xh.dot(x) + self.W_hh.dot(h) + self.bh)
            y = self.W_hy.dot(h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def printSample(self, inputs, h_prev, n):
        if n % 100 == 0:
            sample_ix = self.sample(h_prev, inputs[0], 200)
            txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
            print(f'----\n {(txt, )} \n----')

    def train(self, n=0, p=0):
        smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length

        while True:
            if p+self.seq_length+1 >= len(self.data) or n == 0:
                h_prev = np.zeros((self.hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data
            inputs = [self.char_to_ix[ch] for ch in self.data[p:p+self.seq_length]]
            targets = [self.char_to_ix[ch] for ch in self.data[p+1:p+self.seq_length+1]]

            # sample from the model now and then
            self.printSample(inputs, h_prev, n)

            loss, dW_xh, dW_hh, dW_hy, dbh, dby, h_prev = self.backward(inputs, targets, self.forward(inputs, targets, h_prev))
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            if n % 100 == 0:
                print(f'iter: {n}, loss {smooth_loss}')

            self.updateAdagrad(dW_xh, dW_hh, dW_hy, dbh, dby)

            p += self.seq_length
            n += 1

            if smooth_loss <= 0.05:
                break

    def printWeights(self):
        print(f'W_xh: {self.W_xh}')
        print(f'W_hh: {self.W_hh}')
        print(f'W_hy: {self.W_hy}')
        print(f'bh: {self.bh}')
        print(f'by: {self.by}')




if __name__ == '__main__':
    RNN = RNN()
    RNN.train()
    RNN.printWeights()

        
