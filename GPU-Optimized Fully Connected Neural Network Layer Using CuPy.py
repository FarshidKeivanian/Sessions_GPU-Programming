class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = cp.random.randn(input_size, output_size).astype(cp.float32)
        self.biases = cp.zeros(output_size, dtype=cp.float32)

    def forward(self, x):
        return cp.dot(x, self.weights) + self.biases
