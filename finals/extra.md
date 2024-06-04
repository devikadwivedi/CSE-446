Extra Notes:

### 3. Activation Functions
**Common Activation Functions:**

1. **Sigmoid:**
   $$\[
   \sigma(z) = \frac{1}{1 + e^{-z}}
   \]$$
   - Non-linear, smooth curve
   - Good for probabilistic interpretations 
  **Appropriate For:**
  - **Binary Classification:** The sigmoid function is often used in the output layer of a binary classification network because it maps the output to a range between 0 and 1, representing the probability of the positive class.
  - **Logistic Regression:** In logistic regression models, the sigmoid function is used to model the probability of binary outcomes.

  **Drawbacks:**
  - Vanishing gradients problem: As \(z\) gets very large or very small, the gradient of the sigmoid function approaches zero, slowing down the training.


2. **Tanh:**
   $$\[
   \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
   \]$$
  - Output range: (-1, 1)
  - Non-linear, smooth curve
  - Zero-centered outputs

  **Appropriate For:**
  - **Hidden Layers:** Tanh is often used in hidden layers of neural networks because its output is zero-centered, which can make training faster and convergence easier compared to sigmoid.
  
  **Drawbacks:**
  - Vanishing gradients problem: Similar to the sigmoid function, tanh can also suffer from the vanishing gradient problem.

3. **ReLU (Rectified Linear Unit):**
   $$\[
   \text{ReLU}(z) = \max(0, z)
   \]$$
   - **Benefits:** Computationally efficient, alleviates vanishing gradient problem.
   - **Drawbacks:** Can cause dying ReLU problem (neurons outputting zero).
   - Output range: [0, ∞)
   - Non-linear
   - Computationally efficient
   - **Appropriate For:**
   -   **Hidden Layers in Deep Networks:** ReLU is widely used in hidden layers of deep neural networks because it helps mitigate the vanishing gradient problem and accelerates convergence. It is simple and computationally efficient.
   - **Drawbacks:**
   -    **Dying ReLU Problem:** Neurons can "die" during training, meaning they output zero for any input if they get stuck in the negative side of the function.


4. **Leaky ReLU:**
   $$\[
   \text{Leaky ReLU}(z) = \max(\alpha z, z)
   \]$$
   - **Benefits:** Mitigates dying ReLU problem.
   - **Drawbacks:** Introduces a small slope for negative inputs.
   - 
4. **Leaky ReLU Activation Function**

**Function:**
$$\[ \text{Leaky ReLU}(z) = \max(\alpha z, z) \]$$

**Characteristics:**
- Output range: (-∞, ∞)
- Non-linear
- Allows a small gradient when \(z < 0\)

**Appropriate For:**
- **Hidden Layers in Deep Networks:** Similar to ReLU but helps mitigate the dying ReLU problem by allowing a small, non-zero gradient when the unit is not active.

5. **Softmax Activation Function**

**Function:**
$$\[ \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \]$$

**Characteristics:**
- Output range: (0, 1)
- Outputs sum to 1, representing a probability distribution

**Appropriate For:**
- **Multi-Class Classification:** Used in the output layer of a network for multi-class classification problems. It provides a probability distribution across multiple classes.

6. **Linear Activation Function**

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

### 4. Loss Functions
Loss functions measure the difference between the predicted outputs and true labels. The choice of loss function depends on the task (e.g., regression, classification).

**Common Loss Functions:**

1. **Mean Squared Error (MSE) for Regression:**
   $$\[
   L = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2
   \]$$
   - **Appropriate for:** Regression tasks.
   - **Drawbacks:** Sensitive to outliers.

2. **Cross-Entropy Loss for Classification:**
   $$\[
   L = - \frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
   \]$$
   - **Appropriate for:** Binary classification.
   - **Drawbacks:** Can suffer from numerical instability if $\(\hat{y}_i\)$ is very close to 0 or 1.

3. **Categorical Cross-Entropy for Multi-Class Classification:**
   $$\[
   L = - \frac{1}{m} \sum_{i=1}^m \sum_{c=1}^C y_{i,c} \log(\hat{y}_{i,c})
   \]$$
   - **Appropriate for:** Multi-class classification.
   - **Drawbacks:** Requires one-hot encoded labels.



### 5. Training Process

**Conceptual Explanation:**
Training involves iteratively updating the network's weights to minimize the loss function using gradient-based optimization.

**Detailed Step-by-Step Explanation:**

1. **Initialization:**
   - Initialize weights \(W^{[l]}\) and biases \(b^{[l]}\) randomly or using a specific initialization scheme (e.g., Xavier or He initialization).

2. **Forward Pass:**
   - For each layer \(l\):
     \[
     \mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + b^{[l]}
     \]
     \[
     \mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})
     \]

3. **Compute Loss:**
   - Compute the loss using the appropriate loss function based on the task.

4. **Backpropagation:**
   - Compute gradients of the loss with respect to the weights and biases using the chain rule.
   - For the output layer \(L\):
     \[
     \delta^{[L]} = \frac{\partial L}{\partial \mathbf{a}^{[L]}} \circ \sigma'(\mathbf{z}^{[L]})
     \]
   - For hidden layers \(l = L-1, L-2, \ldots, 1\):
     \[
     \delta^{[l]} = (W^{[l+1]})^T \delta^{[l+1]} \circ \sigma'(\mathbf{z}^{[l]})
     \]
   - Compute gradients:
     \[
     \frac{\partial L}{\partial W^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^T
     \]
     \[
     \frac{\partial L}{\partial b^{[l]}} = \delta^{[l]}
     \]

5. **Update Weights:**
   - Using an optimization algorithm (e.g., Gradient Descent, Adam), update weights and biases:
     \[
     W^{[l]} \leftarrow W^{[l]} - \eta \frac{\partial L}{\partial W^{[l]}}
     \]
     \[
     b^{[l]} \leftarrow b^{[l]} - \eta \frac{\partial L}{\partial b^{[l]}}
     \]

### Example: Training a Simple Neural Network

Consider a neural network with:
- Input layer: 2 neurons
- Hidden layer: 2 neurons
- Output layer: 1 neuron (binary classification)

1. **Initialization:**
   - Initialize \(W^{[1]}\), \(b^{[1]}\), \(W^{[2]}\), \(b^{[2]}\).

2. **Forward Pass:**
   - For layer 1:
     \[
     \mathbf{z}^{[1]} = W^{[1]} \mathbf{x} + b^{[1]}
     \]
     \[
     \mathbf{a}^{[1]} = \text{ReLU}(\mathbf{z}^{[1]})
     \]
   - For layer 2:
     \[
     \mathbf{z}^{[2]} = W^{[2]} \mathbf{a}^{[1]} + b^{[2]}
     \]
     \[
     \hat{y} = \sigma(\mathbf{z}^{[2]})
     \]

3. **Compute Loss:**
   - Using binary cross-entropy loss:
     \[
     L = - (y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}))
     \]

4. **Backpropagation:**
   - For output layer:
     \[
     \delta^{[2]} = \hat{y} - y
     \]
   - For hidden layer:
     \[
     \delta^{[1]} = (W^{[2]})^T \delta^{[2]} \circ \text{ReLU}'(\mathbf{z}^{[1]})
     \]
   - Compute gradients:
     \[
     \frac{\partial L}{\partial W^{[2]}} = \delta^{[2]} (\mathbf{a}^{[1]})^T
     \]
     \[
     \frac{\partial L}{\partial b^{[2]}} = \delta^{[2]}
     \]
     \[
     \frac{\partial L}{\partial W^{[1]}} = \delta^{[1]} (\mathbf{x})^T
     \]
     \[
     \frac{\partial L}{\partial b^{[1]}} = \delta^{[1]}
     \]

5. **

Update Weights:**
   - Update using Gradient Descent:
     \[
     W^{[2]} \leftarrow W^{[2]} - \eta \frac{\partial L}{\partial W^{[2]}}
     \]
     \[
     b^{[2]} \leftarrow b^{[2]} - \eta \frac{\partial L}{\partial b^{[2]}}
     \]
     \[
     W^{[1]} \leftarrow W^{[1]} - \eta \frac{\partial L}{\partial W^{[1]}}
     \]
     \[
     b^{[1]} \leftarrow b^{[1]} - \eta \frac{\partial L}{\partial b^{[1]}}
     \]

This process is repeated for multiple epochs until the loss converges, resulting in a trained neural network.

Feel free to ask any more specific questions or for further clarifications on any part of this explanation!
