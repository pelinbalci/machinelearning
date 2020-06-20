#### Regularization for Sparsity

If we are crossing sparse features, we are going to get a lot of coefficients. This will increase our feature space. 

We end up with NOISY COEFFICIENTS and OVERFITTING. 

⭐️SOLUTION for overfitting : REGULARIZATION. 

⭐QUESTION: Can we regularize in a way that we will also REDUCE OUR MODEL SIZE and MEMORY USAGE?

We would like to 0 out to some weights (not important ones :) ). This will save RAM and helps us with overfitting. 

⭐L0 Regularization: 
'create a regularization term that penalizes the count of non-zero coefficient values in a model. 
Increasing this count would only be justified if there was a sufficient gain in the model's ability to fit the data'
- Penalize you for having nonzero weights. 
- Not convex problem.
- NPHard

⭐Relax L0 problem to L1 Regularization:
- Penalize the sum of absolute weights. 
- Convex Problem
- Encourage sparsity unlike L2. 
- loss(data | model) + λ |w1 + w2 + .... + wn|

L2 and L penalize weights differently:
- L2 penalizes weight^2.
- L1 penalizes |weight|.

Consequently, L2 and L1 have different derivatives:
- The derivative of L2 is 2 * weight.
- The derivative of L1 is k (a constant, whose value is independent of weight).

⭐You can think of the derivative of L2 as a force that removes x% of the weight every time. As Zeno knew, 
even if you remove x percent of a number billions of times, the diminished number will still never quite reach zero.

⭐L1 regularization—penalizing the absolute value of all the weights—turns out to be quite efficient for wide models.

⭐You can think of the derivative of L1 as a force that subtracts some constant from the weight every time.
However, thanks to absolute values, L1 has a discontinuity at 0, which causes subtraction results 
that cross 0 to become zeroed out.

ex:  For example, if subtraction would have forced a weight from +0.1 to -0.2, 
L1 will set the weight to exactly 0.


#### EXERCISE

![Image description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/L1_reg.png)


| Task | Regularization Type | Regularization Rate (lambda) | Test Loss | Train Loss | Test - Train Loss | X1 | X2 | X1^2 | X2^2 | X1X2 | 
|---|----|---- |---| --- | --- | ---- | --- | --- | --- | --- |
| 1 | L2 | 0.1 | 0.188 | 0.094 | 0.094 |-0.26 | -0.16 | 0.0066 | -0.022 | 0.39 |
| 2 | L2 | 0.3 | 0.178 | 0.101 | 0.077 |-0.16 | -0.098 | 0.0085 | -0.024 | 0.29 |
| 3 | L1 | 0.1 | 0.162 | 0.122 | 0.040 | 0 | -0.054 | 0 | -0.022 | 0.28 |
| 4 | L1 | 0.3 | 0.165 | 0.148 | 0.017 | 0 | 0 | 0 | 0 | 0.16 |


- Switching from L2 to L1 regularization dramatically reduces the delta between test loss and training loss.
- Switching from L2 to L1 regularization dampens all of the learned weights.
- Increasing the L1 regularization rate generally dampens the learned weights; however, if the regularization rate goes 
too high, the model can't converge and losses are very high.

- When we increase the regularization rate for L2; trainin loss INCREASES (because we've added a term to loss function) 
while test loss DECREASES. The delta btw test and train DROPS. Weights are lower --> The model complexity is lower. 