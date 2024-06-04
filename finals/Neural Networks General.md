### 1. Architecture of Neural Networks

**Conceptual Explanation:**
A neural network consists of interconnected layers of nodes (neurons).
- **Input Layer:** Receives the input features.
- **Hidden Layers:** Intermediate layers that transform inputs into more abstract features.
- **Output Layer:** Produces the final output of the network.

For a neural network with $\(L\)$ layers (including the input and output layers), each layer $\(l\)$ has $\(n_l\)$ neurons. The weights between layer $\(l\)$ and layer $\(l+1\)$ are represented by a matrix $\(W^{[l]}\)$ of shape $\(n_{l+1} \times n_l\)$, and biases are represented by a vector $\(b^{[l]}\)$ of shape $\(n_{l+1} \times 1\)$.

### 2. Node Notation

**Conceptual Explanation:**
Each node (neuron) in a layer performs a weighted sum of its inputs, adds a bias term, and applies an activation function to produce an output.

**Mathematical Explanation:**
For a given layer $\(l\)$:
- Let $\(\mathbf{a}^{[l-1]}\)$ be the activations from the previous layer.
- The linear transformation is given by:
  $$\[
  \mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + b^{[l]}
  \]$$
- The activation function $\(\sigma\)$ is applied to $\(\mathbf{z}^{[l]}\)$:
  $$\[
  \mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})
  \]$$
