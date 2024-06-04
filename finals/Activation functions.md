### 1. **Sigmoid Activation Function**

**Function:**
$$\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]$$

**Characteristics:**
- Output range: (0, 1)
- Non-linear, smooth curve
- Good for probabilistic interpretations

**Appropriate For:**
- **Binary Classification:** The sigmoid function is often used in the output layer of a binary classification network because it maps the output to a range between 0 and 1, representing the probability of the positive class.
- **Logistic Regression:** In logistic regression models, the sigmoid function is used to model the probability of binary outcomes.

**Drawbacks:**
- Vanishing gradients problem: As \(z\) gets very large or very small, the gradient of the sigmoid function approaches zero, slowing down the training.

### 2. **Tanh Activation Function**

**Function:**
$$\[ \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \]$$

**Characteristics:**
- Output range: (-1, 1)
- Non-linear, smooth curve
- Zero-centered outputs

**Appropriate For:**
- **Hidden Layers:** Tanh is often used in hidden layers of neural networks because its output is zero-centered, which can make training faster and convergence easier compared to sigmoid.

**Drawbacks:**
- Vanishing gradients problem: Similar to the sigmoid function, tanh can also suffer from the vanishing gradient problem.

### 3. **ReLU (Rectified Linear Unit) Activation Function**

**Function:**
$$\[ \text{ReLU}(z) = \max(0, z) \]$$

**Characteristics:**
- Output range: [0, ∞)
- Non-linear
- Computationally efficient

**Appropriate For:**
- **Hidden Layers in Deep Networks:** ReLU is widely used in hidden layers of deep neural networks because it helps mitigate the vanishing gradient problem and accelerates convergence. It is simple and computationally efficient.

**Drawbacks:**
- **Dying ReLU Problem:** Neurons can "die" during training, meaning they output zero for any input if they get stuck in the negative side of the function.

### 4. **Leaky ReLU Activation Function**

**Function:**
$$\[ \text{Leaky ReLU}(z) = \max(\alpha z, z) \]$$

**Characteristics:**
- Output range: (-∞, ∞)
- Non-linear
- Allows a small gradient when \(z < 0\)

**Appropriate For:**
- **Hidden Layers in Deep Networks:** Similar to ReLU but helps mitigate the dying ReLU problem by allowing a small, non-zero gradient when the unit is not active.

### 5. **Softmax Activation Function**

**Function:**
\[ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \]

**Characteristics:**
- Output range: (0, 1)
- Outputs sum to 1, representing a probability distribution

**Appropriate For:**
- **Multi-Class Classification:** Used in the output layer of a network for multi-class classification problems. It provides a probability distribution across multiple classes.

### 6. **Linear Activation Function**

**Function:**
$$\[ \text{linear}(z) = z \]$$

**Characteristics:**
- Output range: (-∞, ∞)
- No non-linearity

**Appropriate For:**
- **Regression Tasks:** Used in the output layer for regression problems where the predicted output is a continuous value.

### Summary

- **Sigmoid:** Best for binary classification outputs.
- **Tanh:** Good for hidden layers, especially when you want zero-centered outputs.
- **ReLU:** Preferred for hidden layers in deep networks due to efficiency and ability to mitigate vanishing gradients.
- **Leaky ReLU:** Used in hidden layers to prevent dying ReLU problem.
- **Softmax:** Ideal for multi-class classification output layers.
- **Linear:** Used in the output layer for regression tasks.

