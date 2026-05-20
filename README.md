# DL_Beginner-Loss_function

# Perceptron Algorithm from Scratch

This project implements the Perceptron Learning Algorithm from scratch using Python and NumPy.

## Overview

The Perceptron is one of the earliest supervised machine learning algorithms used for binary classification. It learns a linear decision boundary by iteratively updating weights whenever a training sample is misclassified.

## Perceptron Model

For an input vector:

x = [x₁, x₂, ..., xₙ]

The perceptron computes:

z = wᵀx + b

where:

- w = weight vector
- b = bias
- z = linear output

Prediction:

```
ŷ = sign(z)
```

Output:

- +1 → Positive Class
- -1 → Negative Class

---

## Perceptron Loss Function

The Perceptron Loss penalizes only misclassified samples.

\[
L(y,z) = \max(0,-yz)
\]

Where:

- y = actual class label (+1 or -1)
- z = wᵀx + b

### Interpretation

#### Correct Classification

If:

\[
yz > 0
\]

then:

\[
L = 0
\]

No update is required.

#### Misclassification

If:

\[
yz < 0
\]

then:

\[
L = -yz
\]

Weights and bias are updated.

---

## Weight Update Rule

When a sample is misclassified:

\[
w = w + \eta yx
\]

\[
b = b + \eta y
\]

where:

- η = learning rate
- y = true label
- x = feature vector

---

## Algorithm

1. Initialize weights and bias.
2. Iterate through training samples.
3. Compute:

   ```
   z = wᵀx + b
   ```

4. Check:

   ```
   y × z < 0
   ```

5. If misclassified:
   - Update weights
   - Update bias
6. Repeat for multiple epochs.

---

## Implementation

```python
def perceptron(x, y):

    w1 = w2 = b = 1
    lr = 0.1

    for epoch in range(1000):

        for i in range(x.shape[0]):

            z = w1 * x[i][0] + w2 * x[i][1] + b

            if z * y[i] < 0:

                w1 = w1 + lr * y[i] * x[i][0]
                w2 = w2 + lr * y[i] * x[i][1]
                b = b + lr * y[i]

    return w1, w2, b
```

---

## Dataset Generation

```python
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    random_state=41,
    hypercube=False,
    class_sep=15
)
```

Convert labels from:

```python
0,1
```

to

```python
-1,+1
```

```python
y = np.where(y == 0, -1, 1)
```

---

## Requirements

```bash
pip install numpy
pip install scikit-learn
```

---

## Results

After training, the perceptron learns:

- Optimal weights
- Bias
- Linear decision boundary

for separating the two classes.

---

## Concepts Covered

- Binary Classification
- Linear Classifier
- Perceptron Learning Algorithm
- Perceptron Loss Function
- Gradient-Free Optimization
- Weight Updates
- Decision Boundary

---

## Author

Laboni Sarkar
M.Tech CSE | Machine Learning & AI Enthusiast
