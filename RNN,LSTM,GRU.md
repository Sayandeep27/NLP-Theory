# Recurrent Neural Networks (RNN), LSTM, and GRU – Complete Guide

Last Updated: 2026

---

## Table of Contents

1. Introduction to Sequential Data
2. What is a Recurrent Neural Network (RNN)?
3. Mathematical Formulation of RNN
4. Problems with Vanilla RNN
5. Long Short-Term Memory (LSTM)
6. LSTM Architecture (Gate-by-Gate Explanation)
7. Mathematical Formulation of LSTM
8. Gated Recurrent Unit (GRU)
9. GRU Architecture (Gate-by-Gate Explanation)
10. Mathematical Formulation of GRU
11. RNN vs LSTM vs GRU (Comparison Table)
12. When to Use What?
13. End-to-End Examples

    * Text Classification
    * Time Series Forecasting
14. Practical Tips & Best Practices
15. Interview Notes & Key Takeaways

---

## 1. Introduction to Sequential Data

Sequential data is data where **order matters**. Unlike tabular data, the meaning of a data point depends on previous data points.

### Examples:

* Text: words in a sentence
* Time series: stock prices, temperature
* Audio: speech signals
* Video: frames over time

Traditional neural networks (ANNs, CNNs) **cannot remember past inputs**. This is where **Recurrent Neural Networks (RNNs)** come in.

---

## 2. What is a Recurrent Neural Network (RNN)?

An **RNN** is a neural network designed to handle sequential data by maintaining a **hidden state** that captures information from previous time steps.

### Key Idea:

> The output at time `t` depends on the input at time `t` **and** the hidden state from time `t-1`.

### Unrolled RNN Representation:

```
Input:  x1 → x2 → x3 → x4
Hidden: h1 → h2 → h3 → h4
Output: y1 → y2 → y3 → y4
```

---

## 3. Mathematical Formulation of RNN

For each time step `t`:

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)

y_t = W_hy · h_t + b_y
```

Where:

* `x_t` = input at time t
* `h_t` = hidden state
* `W` = weight matrices
* `b` = bias

---

## 4. Problems with Vanilla RNN

### 1. Vanishing Gradient Problem

Gradients shrink exponentially during backpropagation through time (BPTT), making it hard to learn long-term dependencies.

### 2. Exploding Gradient Problem

Gradients grow uncontrollably.

### 3. Short-Term Memory

RNNs struggle to remember information far back in the sequence.

**Solution:** LSTM and GRU

---

## 5. Long Short-Term Memory (LSTM)

LSTM is a special type of RNN designed to **remember long-term dependencies**.

### Key Innovation:

* Explicit **memory cell (C_t)**
* Controlled by **gates**

---

## 6. LSTM Architecture (Gate-by-Gate Explanation)

### LSTM Gates:

1. **Forget Gate** – What to forget?
2. **Input Gate** – What to store?
3. **Cell State Update**
4. **Output Gate** – What to output?

### Gate Equations:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)

i_t = σ(W_i · [h_{t-1}, x_t] + b_i)

C~_t = tanh(W_c · [h_{t-1}, x_t] + b_c)

C_t = f_t * C_{t-1} + i_t * C~_t

o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

h_t = o_t * tanh(C_t)
```

### Why LSTM Works:

* Cell state acts like a **conveyor belt**
* Gates regulate information flow

---

## 7. Mathematical Formulation of LSTM

| Component   | Purpose                 |
| ----------- | ----------------------- |
| Forget Gate | Removes irrelevant info |
| Input Gate  | Adds new info           |
| Cell State  | Long-term memory        |
| Output Gate | Controls output         |

---

## 8. Gated Recurrent Unit (GRU)

GRU is a **simplified version of LSTM** with fewer gates.

### Key Differences:

* No separate cell state
* Fewer parameters
* Faster training

---

## 9. GRU Architecture (Gate-by-Gate Explanation)

### GRU Gates:

1. **Update Gate (z_t)** – How much past info to keep?
2. **Reset Gate (r_t)** – How much past info to forget?

### Equations:

```
z_t = σ(W_z · [h_{t-1}, x_t])

r_t = σ(W_r · [h_{t-1}, x_t])

h~_t = tanh(W · [r_t * h_{t-1}, x_t])

h_t = (1 - z_t) * h_{t-1} + z_t * h~_t
```

---

## 10. Mathematical Formulation of GRU

| Gate        | Function            |
| ----------- | ------------------- |
| Update Gate | Controls memory     |
| Reset Gate  | Controls forgetting |

---

## 11. RNN vs LSTM vs GRU (Comparison Table)

| Feature            | RNN  | LSTM | GRU      |
| ------------------ | ---- | ---- | -------- |
| Long-Term Memory   | ❌    | ✅    | ✅        |
| Vanishing Gradient | ❌    | ✅    | ✅        |
| Complexity         | Low  | High | Medium   |
| Training Speed     | Fast | Slow | Faster   |
| Parameters         | Few  | Many | Moderate |

---

## 12. When to Use What?

| Scenario        | Recommended Model |
| --------------- | ----------------- |
| Small dataset   | GRU               |
| Long sequences  | LSTM              |
| Simple patterns | RNN               |
| Limited compute | GRU               |

---

## 13. End-to-End Examples

### Example 1: Text Classification (LSTM)

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential([
    Embedding(input_dim=10000, output_dim=128),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

---

### Example 2: Time Series Forecasting (GRU)

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

model = Sequential([
    GRU(64, return_sequences=False, input_shape=(50, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()
```

---

## 14. Practical Tips & Best Practices

* Normalize time series data
* Use **dropout** to prevent overfitting
* Prefer GRU for faster experiments
* Stack RNN layers carefully
* Use attention for long sequences

---

## 15. Interview Notes & Key Takeaways

* RNNs suffer from vanishing gradients
* LSTM solves long-term dependency problem
* GRU is computationally efficient
* Gates control information flow
* LSTM > GRU > RNN in expressive power

---

## License

MIT License

---

### Author

Sarkar Sayandeep

---

If this README helped you, ⭐ star the repo and share it!
