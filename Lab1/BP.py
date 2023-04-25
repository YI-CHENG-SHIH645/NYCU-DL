import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    labels = list(map(lambda instance: int(instance[0] > instance[1]), pts))

    return pts, np.array(labels).reshape(n, 1)


def generate_xor_easy():
    r = np.arange(0, 1.1, 0.1)
    pts = np.r_[np.c_[r, r], np.c_[r, r[::-1]]]
    pts = np.unique(pts, axis=0)
    labels = [1 if i % 2 == 0 else 0 for i in range(len(pts))]

    return pts, np.array(labels).reshape(21, 1)


def show_results(x, y, y_pred):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, label, title in zip(range(2),
                               [y, y_pred],
                               ['Ground truth', 'Predict result']):
        c = list(map(lambda l: 'r' if l else 'b', label))
        axes[i].set_title(title)
        axes[i].scatter(x[:, 0], x[:, 1], c=c)


def plot_loss_history(loss_history, title):
    ax = plt.subplot()
    ax.plot(loss_history)
    ax.set_title(title)


def sigmoid(v):
    return np.clip(1.0 / (1.0 + np.exp(-v)), 1e-6, 1-1e-6)


def derivative_sigmoid(v):
    return np.multiply(sigmoid(v), 1.0 - sigmoid(v))


def identity(v):
    return v


def derivative_identity(v):
    return v/v


def weights():
    return 'w3', 'b3', 'w2', 'b2', 'w1', 'b1'


class Model:
    def __init__(self, input_dim, h1=5, h2=5, activation='sigmoid'):
        self.w1 = np.random.uniform(size=(input_dim, h1))
        self.b1 = np.random.uniform(size=(h1,))
        self.w2 = np.random.uniform(size=(h1, h2))
        self.b2 = np.random.uniform(size=(h2,))
        self.w3 = np.random.uniform(size=(h2, 1))
        self.b3 = np.random.uniform(size=(1,))
        self.w1_raw_out = None
        self.d_z_w1 = None
        self.w2_raw_out = None
        self.d_z_w2 = None
        self.raw_out = None
        self.d_z_w_out = None
        self.activation = globals()[activation]
        self.der_activation = globals()['derivative_'+activation]

    def forward_pass(self, x):
        self.d_z_w1 = x
        self.w1_raw_out = x @ self.w1 + self.b1
        x = self.activation(self.w1_raw_out)

        self.d_z_w2 = x
        self.w2_raw_out = x @ self.w2 + self.b2
        x = self.activation(self.w2_raw_out)

        self.d_z_w_out = x
        self.raw_out = x @ self.w3 + self.b3
        y_hat = sigmoid(self.raw_out)

        return y_hat

    def backward_pass(self, y, y_hat):
        d_loss_y = -y / y_hat + (1 - y) / (1 - y_hat)
        d_y_z_out = derivative_sigmoid(self.raw_out)
        d_loss_z = d_y_z_out * d_loss_y

        g_loss_w3 = (self.d_z_w_out.T @ d_loss_z) / len(y)
        g_loss_b3 = d_loss_z.mean(0)

        d_loss_z = self.der_activation(self.w2_raw_out) * (d_loss_z @ self.w3.T)

        g_loss_w2 = (self.d_z_w2.T @ d_loss_z) / len(y)
        g_loss_b2 = d_loss_z.mean(0)

        d_loss_z = self.der_activation(self.w1_raw_out) * (d_loss_z @ self.w2.T)

        g_loss_w1 = (self.d_z_w1.T @ d_loss_z) / len(y)
        g_loss_b1 = d_loss_z.mean(0)

        return (g_loss_w3, g_loss_b3,
                g_loss_w2, g_loss_b2,
                g_loss_w1, g_loss_b1)

    def apply_gradients(self, wg, lr=0.01):
        for w, g in wg:
            setattr(self, w, getattr(self, w) - lr*g)


def loss_func(y, y_hat):
    return (-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)).mean()


def train(x, y, epochs, title=None, lr=0.1, activation='sigmoid'):
    print(f"train {title}: ")
    model = Model(x.shape[1], h1=5, h2=5, activation=activation)
    loss_history = []
    for epoch in range(1, epochs+1):
        y_hat = model.forward_pass(x)
        loss = loss_func(y, y_hat)
        loss_history.append(loss)
        if epoch % 5000 == 0:
            y_correct = np.array((y_hat >= 0.5).astype(int) == y)
            print(f'epoch: {epoch} '
                  f'loss: {loss:.3f} '
                  f'acc: {y_correct.sum()/len(y)}')
        grads = model.backward_pass(y, y_hat)
        model.apply_gradients(zip(weights(), grads), lr=lr)
    raw_pred = model.forward_pass(x)
    print(raw_pred)
    y_pred = (raw_pred >= 0.5).astype(int)

    show_results(x, y, y_pred)
    plt.show()

    plot_loss_history(loss_history, title)
    plt.show()


if __name__ == '__main__':
    train(*generate_linear(), epochs=int(1e5), title='linear', lr=0.1, activation='sigmoid')
    train(*generate_xor_easy(), epochs=int(1e5), title='xor', lr=0.1, activation='sigmoid')
