---
layout: post
title:  "Automating Economics to Drive Growth"
author: Tom
categories: [ Articles, Economics, Business, Elasticity of Demand, Python, Regression, OLS]
image: assets/images/elasticity-article.jpg
featured: true
project: false
---


This write up is for measuring the elasticity of demand for individual products in a company's portfolio. Using the results we can optimize pricing to grow revenue. This is especially effective in developed markets where historic data allows for more accuracy results. The math is fairly straightforward, as the function simply using OLS linear regression, and iterate through using a simple python loop.    

Elasticity of demand measures the sensitivity of pricing for a consumer. This measure is invaluable in creating effective pricing strategies. Simply, elasticity of demand is a partial derivative of a demand curve. When you assume a linear demand curve, it is simple as M in y = mx + b. In reality, a demand curve is not linear. It is only assumed to be so when we try to simplify some integral and derivative calculations. To truly calculate elasticity of demand, the partial derivative needs to be calculated and multiple with Delta P/ Delta Q.  

In the supply-demand model, we can easily find a supply curve -  It is the marginal cost curve. The demand model, however, is a little more complicated. The demand curve is defined as the marginal utility curve, but utility is a subjective thing to measure on a case by case basis. If Person A loves cars and Person B doesn't have much interest in cars, then the utility of buying a new car would differ drastically between the two. In the real world, we can't perfectly define a demand curve. We can, however, observe historic behavior to estimate demand.       

So this is typically done using a regression analysis. Lucky for us, data science tools make regression analysis incredibly easy. Where it is the lm() function in R, or Scikit-learn or statsmodels implementations of the analysis in Python. I personally enjoy using Python more, so that is what this script is going to coded in.

First, we import some of the libraries that we will be using. These are incredibly powerful tools that allow us to focus on the assumptions, testing, and take aways. I find that it is perfectly ok to use libraries as long as you understand the underlying math behind the analysis. I would equate libraries to a hammer. You aren't going to go and build one from scratch just to drive a nail into a house. You would just go out and buy one instead. This helps us focus on the task at hand, rather than getting caught up in re-inventing the wheel.


```python
#Import Modules
import pandas as pd
import pyodbc
import numpy as np
import statsmodels.api as sm

```


Our EOD() function filters based on individual product keys and reshapes Pandas series into NumPy columns arrays. WE transform the values into log values. In the end this will change the way we interpret the results from a regular regression. When using a log-log regression the result would be "A X% change in the price would elicit a Beta*X% change in quantity purchased". The model is then fitted and only the parameter we find most important are written into a data frame.


```python
def EOD(FILTER, DF,  MEASURE_ON = 'PROD_KEY'):
    DF = DF.loc[DF[MEASURE_ON] == FILTER, ['UNIT_PRICE' , 'QUANTITY']]

    #col format ,'UNIT_PRICE_2']
    x = DF['UNIT_PRICE'].values.reshape(-1, 1)
    y = DF['QUANTITY'].values.reshape(-1, 1)
    # log-log conversion
    x = np.log(x)
    y = np.log(y)

    data = sm.add_constant(x)
    model = sm.OLS(y, data).fit()

    try:
        A = model.params [1]
        B = model.rsquared
        C = model.pvalues[1]
        D = model.conf_int(alpha=0.5, cols=None)[1][0]#50% confidence interval
        E = model.conf_int(alpha=0.5, cols=None)[1][1]#50% confidence interval
        F = x.shape[0]
        model = None
    except:
        A = 0
        B = 0
        C = 0
        D = 0
        E = 0
        F = 'insufficient'
    return A, B, C, D, E, F

```


```python
#Read your data here
DF = pd.read_csv()

#Data sanity check
DF.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROD_KEY</th>
      <th>UNIT_PRICE</th>
      <th>QUANTITY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>716</td>
      <td>0.01</td>
      <td>25000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>515</td>
      <td>0.01</td>
      <td>8215350.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>732</td>
      <td>0.01</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>316</td>
      <td>0.01</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>148</td>
      <td>0.01</td>
      <td>6.0</td>
    </tr>
  </tbody>
</table>
</div>



Here we are creating a list of product keys that we want to iterate through to find elasticities for.


```python
PRODS = DF['PROD_KEY'].values
PRODS = np.unique(PRODS)
```

This is essentially what the EOD() function is doing, but is only selecting certain parameters from the model.summary().  The add_constant() method adds an column of ones to an array. By default sm.OLS doesn't include an intercept, so needs to be added to preform a legitimate OLS regression. An OLS regression stands for Ordinary Least Squares. In a sentence, It minimizes the sum of squared error for a given model against the observed values.



```python
data = sm.add_constant(x)
model = sm.OLS(y, data).fit()
model.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th>  <td>   0.364</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.364</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.207e+05</td>
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 21 Mar 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>09:48:34</td>     <th>  Log-Likelihood:    </th> <td>-4.4329e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>210705</td>      <th>  AIC:               </th>  <td>8.866e+05</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>210703</td>      <th>  BIC:               </th>  <td>8.866e+05</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    5.2531</td> <td>    0.007</td> <td>  770.181</td> <td> 0.000</td> <td>    5.240</td> <td>    5.266</td>
</tr>
<tr>
  <th>x1</th>    <td>   -0.8229</td> <td>    0.002</td> <td> -347.431</td> <td> 0.000</td> <td>   -0.828</td> <td>   -0.818</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>6021.991</td> <th>  Durbin-Watson:     </th> <td>   1.893</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7332.914</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 0.361</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 3.559</td>  <th>  Cond. No.          </th> <td>    4.89</td>
</tr>
</table>



Here we are iterating through each of the product keys and storing them in a Pandas DataFrame for easy manipulation, export, or analysis. The beta of the each product should be negative, and only in rare cases is the beta zero. In ever rarer cases it can be positive. In which case, your data either lacks enough observations for that product, a weird deal structure created an anomaly or that product is a Giffen or Veblen good. The most latter case is incredibly rare, so I would be relatively confident in writing that possibility off for one of the former cases.    



```python
df = pd.DataFrame(columns=['PROD_KEY', 'BETA', 'PVAL', 'P25', 'P75' , 'R2', 'Records'])
for i in range(PRODS.shape[0]):   
    Beta, Rsquare,Pval, P25, P75, Records = EOD(PRODS[i], DF)
    df.loc[i] = [PRODS[i],Beta, Pval, P25, P75, Rsquare, Records]

```


```python
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROD_KEY</th>
      <th>BETA</th>
      <th>PVAL</th>
      <th>P25</th>
      <th>P75</th>
      <th>R2</th>
      <th>Records</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>144</td>
      <td>-0.917165</td>
      <td>0.086176</td>
      <td>-1.272416</td>
      <td>-0.561914</td>
      <td>0.066926</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>147</td>
      <td>-0.571182</td>
      <td>0.336479</td>
      <td>-0.970016</td>
      <td>-0.172348</td>
      <td>0.038537</td>
      <td>26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>150</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>insufficient</td>
    </tr>

  </tbody>
</table>
</div>



The end result is a list of products and their respective elasticities of demand. From this, we can create optimal pricing strategy. A elasticity less than -1 should get a price decrease, but elasticities greater than -1 should get a price bump. Businesses in developed markets can improve revenue by optimizing to the market they serve.   
