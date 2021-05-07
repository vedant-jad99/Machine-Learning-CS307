# Machine-Learning-CS307
Contains all the tutorials for this course. The following tutorials were implemented:

## Tutorial 1 - Maximum Likelihood Estimation
Task - Do the following experiment for today's Lab:
- Draw 10 sample from Gaussian distribution of mean =5 and variance = 1
- Draw likelihood function for mean between 0 to 10. (Keep var = 1 constant)
- Post the plot here in Google Classroom.

- - -

## Tutorial 2 - Curve Fitting using MLE
Task - For a given dataset: (1,1.2), (2,1.9), (3,3.2)

Find the line which fits the data using maximum likelihood function. Plot the line with the given dataset and post it here in the Google class room. Also create your github account and post the link of the code along with the plot so that I can check the code by clicking on it.

Please take beta = 1; therefore you need to find only w (i.e. intercept and slope only).

- - -

## Tutorial 3 - Bayesian Linear Regression
Task - Plot Figure 3.7 from PRML using your preferable language. Use line of intercept = -0.3 and slope =0.5 with gaussian noise (mean =0; var = 0.04) to generate 10 sample points (x_i, y_i) between -1 and +1 on x axis. For more details refer respective video lecture and PRML.

Hint: You can define 2 dimensional grid uniformly between -1 and +1 on x-aix and y-axis respectively. Then calculate the probability of Gaussian for given mean and variance using library functions. Calculate the un-normalized posterior by multiplying (one-to-one) prior and likelihood grids; hence results in the same dimensional grid with posterior probability. Use heat map to plot the figure. Prior is gaussian distribution with mean = 0 and var = 0.5.

- - -

## Tutorial 4 - Bayesian Linear Regression Continued
Task - Alongwith the previous figure, plot a third column of figures showing randomly chosen points from the posterior distribution and plotting them on data space.

- - -

## Tutorial 5 - Kernel Regression
Task - Code the Kernel Regression in Python.

Dataset: Take 10 points (x_i) from the uniform distribution from 0 to 1, then y_i = Sin(2*pi*x_i) + random noise. Random noise can be draw from normal standard distribution.

PS: You need to approximately reproduce the same figure 6.3 from PRML.

- - -

## Tutorial 6 - Gaussian Process for Regression
Task - For you assignment please use the following code for generating data set (__ is used for indentation). Rest of the code is given in the slides.

 ```
 def dataSet_2():

  __X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
  __Y_train = np.sin(X_train)
  __X_test = np.arange(-5, 5, 0.2).reshape(-1, 1)
  __return X_train,Y_train,X_test
```
For plotting variance and mean use the following code:

```
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):

  __X = X.ravel()
  __mu = mu.ravel()
  __uncertainty = 1.96 * np.sqrt(np.diag(cov))

  __plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
  __plt.plot(X, mu, label='Mean')
  __for i, sample in enumerate(samples):
  ____plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
  __if X_train is not None:
  ____plt.plot(X_train, Y_train, 'rx')
  __plt.legend()
  ```

- - -

## Tutorial 7 - Support Vector Machine (Soft Margin)
Task - Implement soft margin SVM on Moon Dataset usign Gaussian Kernel. The code is given in slides for hard margin.
