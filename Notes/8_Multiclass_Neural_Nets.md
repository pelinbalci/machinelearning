There are two different multiclass solution: 

### One vs All: 

One vs. all provides a way to leverage binary classification. 

Given a classification problem with N possible solutions, a one-vs.-all solution consists of N separate 
binary classifiers—one binary classifier for each possible outcome. 


For example: 

    - a dog picture is given.
    - 5 different recognizers (neurons) might be trained.
    - 4 of them see the image as a negative example (not a dog) and 1 of them sees the image as a positive example (a dog).

1. Is this image an apple? No.
2. Is this image a bear? No.
3. Is this image candy? No.
4. Is this image a dog? Yes.
5. Is this image an egg? No.

This approach is useful when the total number of classes is SMALL.

We can create a significantly more efficient one-vs.-all model with a deep neural network in which each output node 
represents a different class. 

        - Hidden Layers
        - Logits (Sigmoid) Layer: each neuron represents probability of a different class.
        
### Softmax:      

⭐️ Logistic regression produces a decimal between 0 and 1.0. 

For example, a logistic regression output of 0.8 from an email classifier: 

    - 80% chance of an email being spam
    - 20% chance of it being not spam. 
    - The sum of the probabilities of an email being either spam or not spam is 1.0.
    
⭐️ Softmax extends this idea into a multi-class world. 

Softmax assigns decimal probabilities to each class in a multi-class problem. 

Those decimal probabilities must add up to 1.0. 

| Class | Probability | 
|---|----|
| apple | 0.04 |
| bear | 0.06 |
| dog | 0.9 |

⭐️ Note that this is FULL SOFTMAX. It calculates a probability for every possible class

        - Hidden Layers
        - Logits (Softmax) Layer: each neuron represents probability of a different class.
        - Output Layer
        
Mathematical Equation:

        p(y = j | x) = e^(wj * x + bj) / Sum over k( e^wk * x + bk)
        
   
Besides Full Softmax, you can use CANDIDATE SAMPLING. 
It is a training-time optimization in which a probability is calculated:

        - FOR ALL the positive labels, using, for example, softmax, 
        - but only for A RANDOM SAMPLE OF negative labels. 

The idea is that negative classes can learn from less frequent negative reinforcement.

Positive classes always get proper positive reinforcement.

This is indeed observed empirically. 


#### Many Labels

Softmax assumes that each example is a member of EXACTLY 1 class.
 
Some examples, however, can simultaneously be a member of MULTİPLE classes. 
 
For such examples:

    - You may not use Softmax.
    - You must rely on multiple logistic regressions.
    
For example, you have images containing exactly one item—a piece of fruit. 
Softmax can determine the likelihood of that one item being a pear, an orange, an apple, and so on. 

If your examples are images containing different kinds of fruit, 
then you'll have to use multiple logistic regressions instead.