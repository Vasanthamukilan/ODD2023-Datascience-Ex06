# EX-06 FEATURE TRANSFORMATION
### Aim:
To read the given data and perform Feature Transformation process and save the data to a file.
### Explanation:
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.
### Algorithm:
- Step1: Read the given Data.
- Step2: Clean the Data Set using Data Cleaning Process.
- Step3: Apply Feature Transformation techniques to all the features of the data set.
- Step4: Print the transformed features.
### Program:
```
Developed By: Vasanthamukilan M
Register No:2122222301067
```
- Importing libraries and reading csv file:
  ```Python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import statsmodels.api as sm
  import scipy.stats as stats
  from sklearn.preprocessing import QuantileTransformer
  from sklearn.preprocessing import PowerTransformer
  df=pd.read_csv("Data_to_Transform.csv")
  ```
- Basic Information:
  ```Python
  df.head()
  df.info()
  df
  ```
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/a3303641-c88a-4d3c-af14-02bfd504ddf9)

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/977c2604-94eb-4f85-9ae8-871dccdcd504)

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/a835d5a0-f8e9-439d-93ad-a26020a11d09)


- Before Transformation:
  ```Python
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()

  sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
  plt.title("Highly Negative Skew")
  plt.show()

  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
 ![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/0544edf9-26a4-4b07-8819-f73677c3cb6d)

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/8e09d6e0-d6c4-44e5-b80e-52ed11a05998)

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/31e2279d-2003-456c-97f3-d8f222146a0c)

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/96e5e493-6009-4d85-8169-e71e2a7505b6)



- Log Transformation:
  ```Python
  df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  
  df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()
  ```
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/582261b8-d88c-4a72-b566-84c1202f1d5a)
  
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/3a0c6599-8931-4e15-b44e-5394e5e93582)



- Reciprocal Transformation:
  ```Python
  df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/41739d51-5e74-4727-87a1-737da52998c9)



- SquareRoot Transformation:
  ```Python
  df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```

![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/3043ecf4-cc23-4a16-86cd-545e57fe4b9d)


- Power Transformation:
  ```Python
  df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])
  sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
  plt.title("Moderate Positive Skew")
  plt.show()

  transformer=PowerTransformer("yeo-johnson")
  df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate Negative Skew")
  plt.show()
  ```
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/6f508b24-5063-499d-83fb-787722e1905b)


![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/8f378cbf-bf0e-4324-ac23-b17f6506133f)

  
- Quantile Transformation:
  ```Python
  qt = QuantileTransformer(output_distribution = 'normal')
  df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate  Negative Skew")
  plt.show()
  ```
![image](https://github.com/Vasanthamukilan/ODD2023-Datascience-Ex06/assets/119559694/f96ca37d-3a43-4b47-8d64-7ef88d82bbc8)


### Result:  
Thus feature transformation is done for the given dataset.
