+++
date = '2025-01-12T16:29:09+11:00'
draft = false
title = 'Intuitively understanding Decision Trees'
math = true
mathEngine = "mathjax"
+++

# What are decision trees

Decision trees are a simple structure predicting different outcomes (regions) given the 
input satisfies a set of criterion. They generally are poor as a standalone technique however lead
to more powerful drivers of prediction (Random Forest and GBM) through some modification. We will use a regression
tree meaning the response variable $y$ will be continuous in nature. 

The final model will have a set of nodes $i$ which each have a (not necessarily unique) threshold value $t_i$
based on a feature dimension $d_i$. For a new input $x$, we choose the same feature dimension to compare against 
$t_i$, we move down the tree based on the result of the comparison. Finally, when no more decisions are
made and a 'leaf node' (node without any children) has been reached we would like to record the decision 
making process as a region. Each region $R$ is a final output $x$ and records the 
collection of decisions for which $x$ is compared to $t_i$.

These regions are important as they parition the output space into a limited number of results. We therefore
estimate the output of region j using the following 
$$
w_j = \frac{\sum_{n=1}^N y_n \mathbb I (x_n \in R_j)}{\sum_{n=1}^N \mathbb I(x_n \in R_j)}
$$

$\mathbb I (x_n \in R_j)$ represents the indicator function attaining 1 if the n'th row of the input matrix
reaches $R_j$, and $y_n$ is the output of the response variable. The above equation is very intuitive since
summing over all rows in a region is simply a count of how many times our data results in that region being chosen.
Therefore $w_j$ is just the average over all response variables whose input lead to region $R_j$. 

Therefore we define the functional form of a regression tree as 
$$
f(x;\theta) = \sum_{j=1}^J w_j \mathbb I(x \in R_j)
$$
so that $\theta = \{(R_j, w_j) : j = 1 : J\}$ where $J$ is the number of nodes and $\theta$ is the pair of
regions with predicted outputs.


# Fitting decision trees

The next step is more difficult and involves estimating the
feature to split on $j_i$ and a threshold for that feature $t_i$. Ultimately, we would like to minimize the 
non-differentiable loss
$$
L(\theta) = \sum_{n=1}^N l(y_n, f(x_n;\theta))
$$
for an arbitrary loss $l(y_n, f(x_n;\theta))$. minimizing $L(\theta)$ means choosing the appropriate parameters
contained in $\theta$ so that the difference between the actual value $y_n$ and our predicted value 
$f(x_n;\theta)$ is reduced based 
on a chosen metric. For the sake of brevity we will not look at why the loss $L(\theta)$ is non-differentiable, 
however, some motivation is attributed to learning a discrete tree structure meaning finding
an optimal decision tree is np-hard (Cannot be solved efficiently yet). 

An appropriate proxy was found to be 
$$
(j_t,t_i) = \arg \min_{j \in \{1,\cdots,D\}} \min_{t \in T_j} \frac{|D_{i,L}(j,t)|}{|D|} c(D_{i,L}(j,t)) + 
\frac{|D_{i,R}(j,t)|}{|D|} c(D_{i,R}(j,t)) \quad (1.1)
$$
for a cost function $c()$. The above seems somewhat daunting however we will explore the different pieces below.

## Inner expression

The inner expression 
$$
\frac{|D_{i,L}(j,t)|}{|D|} c(D_{i,L}(j,t)) + \frac{|D_{i,R}(j,t)|}{|D|} c(D_{i,R}(j,t))
$$
is characterized by two sets of weighted cost functions. Note that $D_{i,L}(j,t)$ is the set of data points
satisfying the threshold value $x_{n,j} \leq t$. This means that for the n'th example that at $d_i$ we compare
the value of the feature against a threshold. So $|D_{i,L}(j,t)|$ is the number of data points satisfying 
this condition and $|D_{i,R}(j,t)|$ is the number of points satisfying $x_{n,j} > t$. As a clarifying remark
$x_n$ is (generally) a vector containing a number of features whilst $x_{n,j}$ is a data point for an individual
feature. Therefore, if we divide by $|D|$ (total number of data points) we find that the weights of the cost
functions lie in the interval $[0,1]$. 

The cost function $c()$ is chosen by us and is generally substituted for the Mean Squared Error (MSE)
$$
c(D_i) = \frac{1}{|D|} \sum_{n \in D_i} (y_n - \overline{y})^2
$$
where $\overline{y}$ is the mean of response variables reaching node i. 

Put together we have a weighted summation of cost functions. Each cost function represents the magnitude of
difference from the predicted to the average class label for each node depending on whether we move left or
right along the node. 

## Choosing a threshold and decision feature

$\min_{t \in T_j}$ attempts to find the best threshold for each feature j. $T_j$ is written as the set of values 
$\{x_{nj}\}$ (example for the n'th row and j'th feature), a decision rule can then be set on a sorted set of 
unique values. We
generally want binary splits to avoid data fragmentation (splitting the data into too many nodes), therefore
a range can be chosen appropriately for each feature j. So for example if we have $T_j$ = \{5, 10, 15 \} then
we can consider thresholds $t_j < 5$, or $t_j < 10$ etc. After this we choose the best feature dimension j
minimizing equation 1.1.

Ultimately, we 'greedily' choose a threshold for each feature then choose $j$ which appropriately minimizes
the weighted sum of costs overall. Note that this only finds the pair $(j_i, t_i)$ meaning we must minimize
for every node i. 

## How many nodes?

If we let the tree grow without restriction then a training error of 0 can be achieved. Eventually, we attain
a region for each data point meaning the model will overfit. Pruning, or more specifically post-pruning (as 
opposed to pre-pruning) allows the decision tree to grow to max depth then remove branches until the model is not
overfitting anymore. Multiple means for appropriate pruning exist and we can 
specifically use Cost complexity pruning 
which chooses some subtree t minimizing a condition and adding to a new tree. Eventually, the tree with the
best accuracy is chosen. As an alternative pre-pruning can be used which simply stops execution based on some
heuristic, for example if the number of data points in a node becomes too small.


# Why a decision tree?

Decision trees are generally implemented because of automatic variable selection, ease to interpret
robustness and no need for standardization. They are great as a basic model that may lead to insights by
questioning why a threshold and feature for a given node was chosen. 

As with all models there are plenty of drawbacks that can be interpreted from the math. Of course the main
one being that a greedy process is taken to determine the thresholds and decision rules. Namely, we cannot
optimize $L(\theta)$ and so the chosen proxy uses a greedy procedure for threshold and feature selection without
considering what will occur at the next node. Accuracy is hit, reinforcing the aforementioned point that decision
trees are poor as a standalone model. 

# Conclusions

Provided above is a bare bones explanation of the inner workings regarding a decision tree. There are multiple
other processes which can also be used to further increase the accuracy of decision trees which have not been
presented here. Plenty of resources exist to use decision trees for classification and have therefore
not been presented but a general adaptation would be to reform the cost function from MSE to the Gini index. 
Ultimately, the above hopes to give a greater intuition on the inner workings of decision trees and how they 
come to life in the presence of data.