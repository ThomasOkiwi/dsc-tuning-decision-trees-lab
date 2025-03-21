# Hyperparameter Tuning and Pruning in Decision Trees
## Introduction
Hyperparameter tuning relates to how we sample candidate model architectures from the space of all possible hyperparameter values. This is often referred to as searching the hyperparameter space for the optimum values. In this lesson, we'll look at some of the key hyperparameters for decision trees and how they affect the learning and prediction processes.

## Objectives
Identify the role of pruning while training decision trees
List the different hyperparameters for tuning decision trees
Hyperparameter Optimization
In machine learning, a hyperparameter is a parameter whose value is set before the learning process begins.

By contrast, the values of model parameters are derived via training as we have seen previously. Different model training algorithms require different hyperparameters, some simple algorithms (such as ordinary least squares regression) require none. Given these hyperparameters, the training algorithm learns the parameters from the data. For instance, Lasso is an algorithm that adds a regularization hyperparameter to ordinary least squares regression, which has to be set before estimating the parameters through the training algorithm.

In this lesson, we'll look at these sorts of optimizations in the context of decision trees and see how these can affect the predictive performance as well as the computational complexity of the tree.

## Tree pruning
Now that we know how to grow a decision tree using Python and scikit-learn, let's move on and practice optimizing a classifier. We can tweak a few parameters in the decision tree algorithm before the actual learning takes place.

A decision tree, grown beyond a certain level of complexity leads to overfitting. If we grow our tree and carry on using poor predictors that don't have any impact on the accuracy, we will eventually a) slow down the learning, and b) cause overfitting. Different tree pruning parameters can adjust the amount of overfitting or underfitting in order to optimize for increased accuracy, precision, and/or recall.

This process of trimming decision trees to optimize the learning process is called "tree pruning".

We can prune our trees using:

Maximum depth: Reduce the depth of the tree to build a generalized tree. Set the depth of the tree to 3, 5, 10 depending after verification on test data

Minimum samples leaf with split: Restrict the size of sample leaf

Minimum leaf sample size: Size in terminal nodes can be fixed to 30, 100, 300 or 5% of total

Maximum leaf nodes: Reduce the number of leaf nodes

Maximum features: Maximum number of features to consider when splitting a node

Let's look at a few hyperparameters and learn about their impact on classifier performance:

### max_depth
The parameter for decision trees that we normally tune first is max_depth. This parameter indicates how deep we want our tree to be. If the tree is too deep, it means we are creating a large number of splits in the parameter space and capturing more information about underlying data. This may result in overfitting as it will lead to learning granular information from given data, which makes it difficult for our model to generalize on unseen data. Generally speaking, a low training error but a large testing error is a strong indication of this.

If, on the other hand, the tree is too shallow, we may run into underfitting, i.e., we are not learning enough information about the data and the accuracy of the model stays low for both the test and training samples. The following example shows the training and test AUC scores for a decision tree with depths ranging from 1 to 32.

![image](https://github.com/user-attachments/assets/78c93137-b61c-48b8-9951-d84cba1336aa)

In the above example, we see that as the tree depth increases, our validation/test accuracy starts to go down after a depth of around 4. But with even greater depths, the training accuracy keeps on rising, as the classifier learns more information from the data. However this information can not be mapped onto unseen examples, hence the validation accuracy falls down constantly. Finding the sweet spot (e.g. depth = 4) in this case would be the first hyperparameter that we need to tune.

## min_samples_split
The hyperparameter min_samples_split is used to set the minimum number of samples required to split an internal node. This can vary between two extremes, i.e., considering only one sample at each node vs. considering all of the samples at each node - for a given attribute.

When we increase this parameter value, the tree becomes more constrained as it has to consider more samples at each node. Here we will vary the parameter from 10% to 100% of the samples.
![image](https://github.com/user-attachments/assets/c84c6c3b-2d5e-4026-9f40-46bf18e354f6)



In the above plot, we see that the training and test accuracy stabilize at a certain minimum sample split size, and stays the same even if we carry on increasing the size of the split. This means that we will have a complex model, with similar accuracy than a much simpler model could potentially exhibit. Therefore, it is imperative that we try to identify the optimal sample size during the training phase.

Note: max_depth and min_samples_split are also both related to the computational cost involved with growing the tree. Large values for these parameters can create complex, dense, and long trees. For large datasets, it may become extremely time-consuming to use default values.

## min_samples_leaf
This hyperparameter is used to identify the minimum number of samples that we want a leaf node to contain. When this minimum size is achieved at a node, it does not get split any further. This parameter is similar to min_samples_splits, however, this describes the minimum number of samples at the leaves, the base of the tree.

![image](https://github.com/user-attachments/assets/090e7444-9d92-47c5-835d-fb8887572234)


The above plot shows the impact of this parameter on the accuracy of the classifier. We see that increasing this parameter value after an optimal point reduces accuracy. That is due to underfitting again, as keeping too many samples in our leaf nodes means that there is still a high level of uncertainty in the data.

The main difference between the two is that min_samples_leaf guarantees a minimum number of samples in a leaf, while min_samples_split can create arbitrary small leaves, though min_samples_split is more common in practice. These two hyperparameters make the distinction between a leaf (terminal/external node) and an internal node. An internal node will have further splits (also called children), while a leaf is by definition a node without any children (without any further splits).

For instance, if min_samples_split = 5, and there are 7 samples at an internal node, then the split is allowed. But let's say the split results in two leaves, one with 1 sample, and another with 6 samples. If min_samples_leaf = 2, then the split won't be allowed (even if the internal node has 7 samples) because one of the leaves resulted will have less than the minimum number of samples required to be at a leaf node.

## Are there more hyperparameters?
Yes, there are! Scikit-learn offers a number of other hyperparameters for further fine-tuning the learning process. Consult the official docLinks to an external site. to look at them in detail. The hyperparameters mentioned here are directly related to the complexity which may arise in decision trees and are normally tuned when growing trees. We'll shortly see this in action with a real dataset.

Additional Resources
Overview of hyperparameter tuningLinks to an external site.
Demystifying hyperparameter tuningLinks to an external site.
Pruning decision treesLinks to an external site.


# Hyperparameter Tuning and Pruning in Decision Trees - Lab

## Introduction

In this lab, you will use the titanic dataset to see the impact of tree pruning and hyperparameter tuning on the predictive performance of a decision tree classifier. Pruning reduces the size of decision trees by removing nodes of the tree that do not provide much predictive power to classify instances. Decision trees are the most susceptible out of all the machine learning algorithms to overfitting and effective pruning can reduce this likelihood. 

## Objectives

In this lab you will: 

- Determine the optimal hyperparameters for a decision tree model and evaluate the model performance

## Import necessary libraries

Let's first import the libraries you'll need for this lab. 


```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
plt.style.use('seaborn-v0_8-darkgrid')
```

## Import the data

The titanic dataset, available in `'titanic.csv'`, is all cleaned up and preprocessed for you so that you can focus on pruning and optimization. Import the dataset and print the first five rows of the data: 


```python
# Import the data
df = None

```

## Create training and test sets

- Assign the `'Survived'` column to `y` 
- Drop the `'Survived'` and `'PassengerId'` columns from `df`, and assign the resulting DataFrame to `X` 
- Split `X` and `y` into training and test sets. Assign 30% to the test set and set the `random_state` to `SEED` 


```python
# Create X and y 
y = None
X = None

# Split into training and test sets
SEED = 1
X_train, X_test, y_train, y_test = None
```

## Train a vanilla classifier

__Note:__ The term "vanilla" is used for a machine learning algorithm with its default settings (no tweaking/tuning).

- Instantiate a decision tree 
  - Use the `'entropy'` criterion and set the `random_state` to `SEED` 
- Fit this classifier to the training data 


```python
# Train the classifier using training data
dt = None

```

## Make predictions 
- Create a set of predictions using the test set 
- Using `y_test` and `y_pred`, calculate the AUC (Area under the curve) to check the predictive performance


```python
# Make predictions using test set 
y_pred = None

# Check the AUC of predictions
false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
roc_auc
```

## Maximum Tree Depth

Let's first check for the best depth parameter for our decision tree: 

- Create an array for `max_depth` values ranging from 1 - 32  
- In a loop, train the classifier for each depth value (32 runs) 
- Calculate the training and test AUC for each run 
- Plot a graph to show under/overfitting and the optimal value 
- Interpret the results 


```python
# Identify the optimal tree depth for given data

```


```python
# Your observations here 
```

## Minimum Sample Split

Now check for the best `min_samples_splits` parameter for our decision tree 

- Create an array for `min_sample_splits` values ranging from 0.1 - 1 with an increment of 0.1 
- In a loop, train the classifier for each `min_samples_splits` value (10 runs) 
- Calculate the training and test AUC for each run 
- Plot a graph to show under/overfitting and the optimal value 
- Interpret the results


```python
# Identify the optimal min-samples-split for given data

```


```python
# Your observations here
```

## Minimum Sample Leafs

Now check for the best `min_samples_leafs` parameter value for our decision tree 

- Create an array for `min_samples_leafs` values ranging from 0.1 - 0.5 with an increment of 0.1 
- In a loop, train the classifier for each `min_samples_leafs` value (5 runs) 
- Calculate the training and test AUC for each run 
- Plot a graph to show under/overfitting and the optimal value 
- Interpret the results


```python
# Calculate the optimal value for minimum sample leafs

```


```python
# Your observations here 

```

## Maximum Features

Now check for the best `max_features` parameter value for our decision tree 

- Create an array for `max_features` values ranging from 1 - 12 (1 feature vs all)
- In a loop, train the classifier for each `max_features` value (12 runs) 
- Calculate the training and test AUC for each run 
- Plot a graph to show under/overfitting and the optimal value 
- Interpret the results


```python
# Find the best value for optimal maximum feature size

```


```python
# Your observations here
```

## Re-train the classifier with chosen values

Now we will use the best values from each training phase above and feed it back to our classifier. Then we can see if there is any improvement in predictive performance. 

- Train the classifier with the optimal values identified 
- Compare the AUC of the new model with the earlier vanilla decision tree AUC 
- Interpret the results of the comparison


```python
# Train a classifier with optimal values identified above
dt = None


false_positive_rate, true_positive_rate, thresholds = None
roc_auc = None
roc_auc
```


```python
# Your observations here
```

In order to address the issue of a baseline classifier performing better than a tuned one like this, a more-sophisticated technique is called a "grid search" and this will be introduced in a future lesson.

## Summary 

In this lesson, we looked at tuning a decision tree classifier in order to avoid overfitting and increasing the generalization capabilities of the classifier. For the titanic dataset, we see that identifying optimal parameter values can result in some improvements towards predictions. This idea will be exploited further in upcoming lessons and labs. 
