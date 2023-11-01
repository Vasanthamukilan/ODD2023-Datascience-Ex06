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
![277957540-8174e1f4-d724-436b-8f33-0fa20413cddc](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/c4357491-04c7-4608-a958-112237c2433d)

![277957560-a0687a96-21ff-44e2-a19a-a2aaf7d6753d](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/056d7173-d2dc-4a85-b178-1e207091cdb2)
![277957584-132a6dd0-f447-4907-b2b9-7f342e3a8ca4](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/28b2c495-8b37-4a7e-a9ef-c14ada3361c1)

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
  ![277957889-eac11f2a-51cd-4eb0-a348-757a01b48102](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/3a8df91c-cd00-43f6-a7ef-a767ae36a256)

![277957920-63728e2c-5c24-49f5-9865-014c1335304d](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/f362af6c-caf7-420d-8fd7-76e2293b2871)
![277957940-53cdbe49-62be-4158-96f8-cf7d3f1f94ea](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/ca5e2c05-e172-42a0-b896-f95cf47186d8)
![277957956-1f2338b7-8976-4a38-a2fe-a76f99b4ff0b](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/d0263769-611d-4e32-b02e-678f2b1888d6)


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
  ![277958134-567a14c2-975d-4022-ae7e-48e9643d67b8](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/410478dd-f7f7-4ce8-be0e-e1639b4a7822)

![277958165-91cb3661-adb6-4726-901a-9453db9ec65c](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/2d6c85c4-768f-402d-828a-3ad3bfed92ca)


- Reciprocal Transformation:
  ```Python
  df['Highly Positive Skew'] = 1/df['Highly Positive Skew']
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```
![277958317-5e40b4fc-cd70-435b-ab0f-55a6df660225](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/6408c451-246d-45b7-a937-1552db90bbc0)


- SquareRoot Transformation:
  ```Python
  df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)
  sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
  plt.title("Highly Positive Skew")
  plt.show()
  ```

![277958383-9adbe3af-b0b8-41e7-880f-a29fb963951f](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/1de7c66e-2a16-4d53-8e80-fa8c6b18c8ec)

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
 ![277958457-1fc1a6f8-9403-496d-bdb4-6ec5e24d250e](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/046ab531-9f77-4aa4-ae74-f7b3b97d44e8)


![277958480-fdde097f-af98-4bfa-bcc7-1f33b5672a6e](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/6a646196-555a-498a-b31b-b4017a3ca7be)

  
- Quantile Transformation:
  ```Python
  qt = QuantileTransformer(output_distribution = 'normal')
  df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
  sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
  plt.title("Moderate  Negative Skew")
  plt.show()
  ```
![277958544-00458c10-dacb-4fed-9f0f-34a411f7977d](https://github.com/KRISHNARAJ-D/ODD2023-Datascience-Ex06/assets/119559695/3321ee2f-7fc2-4812-8aea-74659a02752a)

### Result:  
Thus feature transformation is done for the given dataset.
