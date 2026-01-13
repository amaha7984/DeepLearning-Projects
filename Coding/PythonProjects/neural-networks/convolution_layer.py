import numpy as np

class conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.weights = 0.1 * np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.biases = np.zeros(out_channels)

    def forward(self, X):
        # X: (B, C, H, W)

        batch_size, channels_in, H, W = X.shape
        K = self.kernel_size
        S = self.stride
        p = self.padding

        C_out = self.weights.shape[0] #number of kernels

        if p > 0:
            X = np.pad(X, ((0,0), (0,0), (p, p), (p, p)), mode = 'constant')
        
        # O = ((W - F + 2 * P)/S) + 1
        output_height = ((X.shape[2] - K)//S) + 1
        output_width = ((X.shape[3] - K)//S) + 1

        output = np.zeros((batch_size, C_out, output_height, output_width))

        for b in range(batch_size):
            for c_out in range(C_out): #for getting kernels
                for i in range(output_height):
                    for j in range(output_width):
                        row = i * S
                        col = j * S
                        patch = X[b, :, row: row+K, col: col+K]
                        kernel = self.weights[c_out]
                        output[b, c_out, i, j] = np.sum(patch * kernel) + self.biases[c_out]

        self.output = output



