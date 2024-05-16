# Autoregressive Models, Maximum Likelihood Estimation Topic Summary 
Barış Sarper Tezcan, Furkan Genç
CENG796 Deep Generative Models

Autoregressive Models, Maximum Likelihood Estimation Topic Summary Outline
1.	Introduction to Autoregressive Models
•	What are autoregressive models?

## Introduction to Autoregressive Models

### What are autoregressive models?
Autoregressive models are a type of generative model used for sequential data, where the future values of the series are predicted based on its past values. These models generate data by sequentially predicting each value in the series conditioned on the previous values.

In the context of generative models, autoregressive models can be extended to generate sequences of data points, such as text, audio, or even images. By modeling the probability distribution of each data point conditioned on the previous points, these models can generate highly realistic sequences.

### Motivating Example: MNIST

The MNIST dataset, which consists of handwritten digits, can be used as a motivating example to explain the concept of autoregressive generative models.

Given the dataset `D` of binarized MNIST images, each image has `n = 28 x 28 = 784` pixels. Each pixel can either be black (`0`) or white (`1`).

The goal is to learn a probability distribution $p(x) = p(x_1, x_2, ..., x_{784})$ over `x` in $\{0, 1\}^{784}$ such that when `x` is drawn from `p(x)`, it looks like a digit. In other words, we want to generate new images that resemble the original handwritten digits.

This process is done in two steps:

1. **Parameterize a model family $\{ p_θ(x), \theta \in \Theta\}$:** This involves defining a model where the probability of each pixel is conditioned on the previous pixels. This means that the value of each pixel is dependent on the values of the pixels that came before it. This is the essence of an autoregressive model, where the output is a function of its own previous values.

2. **Search for model parameters $\theta$ based on training data `D`:** This involves optimizing the parameters to best fit the observed data, typically using methods like maximum likelihood estimation. This step is about training the model on the dataset to find the best parameters that make the model generate images as close as possible to the original ones.


### Chain Rule Factorization
In autoregressive models, the joint probability distribution of a sequence of variables is factorized into a product of conditional probabilities using the chain rule of probability. This factorization allows us to model the complex dependencies between variables in a sequential manner.

#### Why Use Chain Rule Factorization?
The chain rule factorization is used in autoregressive models because it provides a systematic way to decompose the joint distribution of a sequence into simpler, conditional distributions. This decomposition allows us to generate each value in the sequence one at a time, conditioned on the previous values.

Given a sequence of variables $x = (x_1, x_2, ..., x_n)$, the joint probability `p(x)` can be factorized as:
$$p(x) = p(x_1) * p(x_2 | x_1) * p(x_3 | x_1, x_2) * ... * p(x_n | x_1, x_2, ..., x_{n-1}) $$

This factorization makes it feasible to model and generate sequences by sequentially sampling each variable conditioned on the previously generated variables.

### Number of Parameters

The number of parameters required to model the joint distribution depends on the number of conditional distributions we need to learn. For each variable $x_i$, we need to estimate the conditional probability $p(x_i \mid x_1, x_2, \ldots, x_{i-1})$.

1. **For $x_1$:**
   - $p(x_1)$ requires 1 parameter (since it is unconditional).

2. **For $x_2$:**
   - $p(x_2 \mid x_1)$ requires 2 parameters (since $x_2$ can take two values, 0 or 1, given $x_1$).

3. **For $x_3$:**
   - $p(x_3 \mid x_1, x_2)$ requires 4 parameters (since $x_3$ can take two values given each combination of $x_1$ and $x_2$).

4. **General Case for $x_i$:**
   - $p(x_i \mid x_1, x_2, \ldots, x_{i-1})$ requires $2^{i-1}$ parameters.

In the image, this is illustrated using tables that show the conditional probabilities for each variable based on the preceding variables.

### Total Number of Parameters

To find the total number of parameters required to model the entire sequence, we sum the parameters for each conditional distribution:

$$ \text{Total Parameters} = 1 + 2 + 4 + \ldots + 2^{n-1} $$

This is a geometric series with the first term $a = 1$ and the common ratio $r = 2$. The sum of the first $n$ terms of a geometric series is given by:

$$ S_n = a \frac{r^n - 1}{r - 1} $$

For $n = 784$ (since we have 784 pixels):

$$ \text{Total Parameters} = \frac{2^{784} - 1}{2 - 1} = 2^{n} - 1 $$

### Implications

- **Exponential Growth**: The number of parameters grows exponentially with the number of variables. This can make the model very complex and computationally expensive.
- **Scalability**: For large sequences, such as images with many pixels, this exponential growth can become impractical. For instance, for a 28x28 image, the number of parameters would be $2^{784} - 1$, which is practically impossible to learn.



#### Applications in Generative AI
- **Text Generation**: Autoregressive models like GPT (Generative Pre-trained Transformer) predict the next word in a sequence, conditioned on the previous words.
- **Image Generation**: PixelRNN and PixelCNN are autoregressive models that generate images pixel by pixel.
- **Music Generation**: Models like WaveNet generate audio samples in an autoregressive manner.

### Advantages of Autoregressive Models
- **Flexibility**: Can model complex dependencies in sequential data.
- **High-Quality Outputs**: Often produce high-quality and coherent sequences.
- **Interpretability**: The coefficients \( \phi_i \) provide insight into the influence of past values.

### Challenges
- **Computationally Intensive**: Sequential prediction can be slow and computationally expensive.
- **Dependency on Previous Values**: Errors can propagate through the sequence, affecting later predictions.

### Conclusion
Autoregressive models play a crucial role in generative modeling by leveraging past data points to generate realistic and coherent sequences. Understanding their principles and applications is essential for advancing generative AI techniques.


---

### References
- [Wikipedia: Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model)
- [Introduction to Time Series Analysis](https://example.com/introduction-to-time-series-analysis)



2.	Structure of Autoregressive Models
•	Chain Rule Factorization
•	Bayesian Networks vs Neural Models Comparison (emphasizing the lack of conditional independence assumptions in neural models)
3.	How to model Autoregressive Models?
•	Fully Visible Sigmoid Belief Networks (FVSBN) (Contrast between parallelizable likelihood evaluation and sequential data generation)
•	NADE (Neural Autoregressive Density Estimation)
•	General discrete distributions (How to model the density of an RGB image as an autoregressive model?)
•	Autoregressive models vs. Autoencoders (How to model an autoregressive model as an autoencoder?)
4.	Case Study: Pixel CNN (Usage of CNN’s (masked convolution) in autoregressive models)
5.	Summary of Autoregressive Models (Pros and Cons)


6.	Learning a Generative Model
•	Goal of Learning 
•	Learning as Density Estimation
7.	Maximum Likelihood Estimation (MLE)
•	KL Divergence (Formulation, Jensen's inequality, briefly mention some of the interpretations of KL divergence)
•	Relation of KL divergence and expected log-likelihood (minimizing KL divergence is equivalent to maximizing the expected log-likelihood, Monte Carlo estimation)
7.1  Finding optimal parameters with MLE
•	Analytical Derivation (Coin Example)
•	MLE Learning with Gradient Descent in Neural Models
7.2 Empirical Risk and Overfitting (optional)
8. Conclusion (recap of the key points of the concepts being covered)
