## Autoregressive Models, Maximum Likelihood Estimation Topic Summary 
Barış Sarper Tezcan, Furkan Genç
CENG796 Deep Generative Models

Autoregressive Models, Maximum Likelihood Estimation Topic Summary Outline
1.	Introduction to Autoregressive Models
•	What are autoregressive models?
•	Motivating Example MNIST: Two-step process of Autoregressive Models (How to model and how to learn?)
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
