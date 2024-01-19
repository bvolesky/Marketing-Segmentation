# Why?
Marketing Segmentation enables a business to understand its ideal customers and tailor products to meet the specific needs, behaviors, and concerns of different customer groups. This approach allows for more efficient marketing, focusing efforts on segments most likely to purchase a product, rather than marketing to all customers indiscriminately.

## Goal
Need to perform clustering to summarize customer segments.

## Learnings
In this project, I significantly expanded my data science skills by effectively combining various techniques to extract meaningful insights. A crucial lesson was the importance of clean, well-structured data, as it is foundational for accurate analysis. Ensuring the data is in the correct format, with sensible naming and thorough validation, is essential to avoid problems in subsequent processes.

I also gained proficiency in feature engineering, learning to create new variables for correlation analysis and plotting. Understanding the impact of outliers on results and mastering data preprocessing methods like labeling, scaling, and dimensionality reduction further enhanced my analytical abilities.

A key discovery was the use of clustering, specifically employing the elbow method to determine the optimal number of clusters and grouping similar data points. This approach was instrumental in segment profiling, revealing distinct patterns based on income and spending habits. By labeling these clusters accurately, I could assess their significance and the effectiveness of targeted marketing campaigns.

The project culminated in combining age and generational data to understand each cluster's behavior and preferences, including family composition and comparative analysis. This experience was incredibly valuable, and I'm eager to apply these data science techniques in future endeavors.

## Imports


```python
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt, numpy as np
```


```python
# Load the data into a pandas DataFrame for proper data wranglin'
df = pd.read_csv('marketing_campaign.csv', delimiter='\t')
print(f'There are {len(df)} rows in this dataset')
df.head()
```

    There are 2240 rows in this dataset
    




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>04-09-2012</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>08-03-2014</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>21-08-2013</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>10-02-2014</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>19-01-2014</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



## Cleaning the Data
Ok, so the data looks pretty good already - let's clean the data at the tables info


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2240 non-null   int64  
     1   Year_Birth           2240 non-null   int64  
     2   Education            2240 non-null   object 
     3   Marital_Status       2240 non-null   object 
     4   Income               2216 non-null   float64
     5   Kidhome              2240 non-null   int64  
     6   Teenhome             2240 non-null   int64  
     7   Dt_Customer          2240 non-null   object 
     8   Recency              2240 non-null   int64  
     9   MntWines             2240 non-null   int64  
     10  MntFruits            2240 non-null   int64  
     11  MntMeatProducts      2240 non-null   int64  
     12  MntFishProducts      2240 non-null   int64  
     13  MntSweetProducts     2240 non-null   int64  
     14  MntGoldProds         2240 non-null   int64  
     15  NumDealsPurchases    2240 non-null   int64  
     16  NumWebPurchases      2240 non-null   int64  
     17  NumCatalogPurchases  2240 non-null   int64  
     18  NumStorePurchases    2240 non-null   int64  
     19  NumWebVisitsMonth    2240 non-null   int64  
     20  AcceptedCmp3         2240 non-null   int64  
     21  AcceptedCmp4         2240 non-null   int64  
     22  AcceptedCmp5         2240 non-null   int64  
     23  AcceptedCmp1         2240 non-null   int64  
     24  AcceptedCmp2         2240 non-null   int64  
     25  Complain             2240 non-null   int64  
     26  Z_CostContact        2240 non-null   int64  
     27  Z_Revenue            2240 non-null   int64  
     28  Response             2240 non-null   int64  
    dtypes: float64(1), int64(25), object(3)
    memory usage: 507.6+ KB
    

**From the info above I can see that**:
- Columns are named differently than the data dictionary
- Columns are in a different order than the data dictionary
- Income has some null values
- Dt_Customer is not currently stored as a date type
- Some integers like can be stored as booleans like COMPLAINED, and the CAMPAIGN fields
- We should add some additional features from existing columns like Age

First, let's rename the columns to match our downstream column requirements.


```python
# Define the mapping of old column names to new names
column_mapping = {
    'ID': 'ID',
    'Year_Birth': 'BIRTH_YEAR',
    'Education': 'EDUCATION',
    'Marital_Status': 'MARITAL_STATUS',
    'Income': 'INCOME',
    'Kidhome': 'CHILDREN_AT_HOME',
    'Teenhome': 'TEENS_AT_HOME',
    'Dt_Customer': 'START_DATE',
    'Recency': 'DAYS_SINCE_PURCHASE',
    'MntWines': 'WINE',
    'MntFruits': 'FRUIT',
    'MntMeatProducts': 'MEAT',
    'MntFishProducts': 'FISH',
    'MntSweetProducts': 'SWEETS',
    'MntGoldProds': 'GOLD',
    'NumDealsPurchases': 'DISCOUNTED_PURCHASES',
    'NumWebPurchases': 'WEBSITE_PURCHASES',
    'NumCatalogPurchases': 'CATALOG_PURCHASES',
    'NumStorePurchases': 'STORE_PURCHASES',
    'NumWebVisitsMonth': 'WEBSITE_VISITS',
    'AcceptedCmp1': 'CAMPAIGN_1',
    'AcceptedCmp2': 'CAMPAIGN_2',
    'AcceptedCmp3': 'CAMPAIGN_3',
    'AcceptedCmp4': 'CAMPAIGN_4',
    'AcceptedCmp5': 'CAMPAIGN_5',
    'Response': 'ACCEPTED_LAST_CAMPAIGN',
    'Complain': 'COMPLAINED'
}

# Rename the columns
df.rename(columns=column_mapping, inplace=True)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 29 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   ID                      2240 non-null   int64  
     1   BIRTH_YEAR              2240 non-null   int64  
     2   EDUCATION               2240 non-null   object 
     3   MARITAL_STATUS          2240 non-null   object 
     4   INCOME                  2216 non-null   float64
     5   CHILDREN_AT_HOME        2240 non-null   int64  
     6   TEENS_AT_HOME           2240 non-null   int64  
     7   START_DATE              2240 non-null   object 
     8   DAYS_SINCE_PURCHASE     2240 non-null   int64  
     9   WINE                    2240 non-null   int64  
     10  FRUIT                   2240 non-null   int64  
     11  MEAT                    2240 non-null   int64  
     12  FISH                    2240 non-null   int64  
     13  SWEETS                  2240 non-null   int64  
     14  GOLD                    2240 non-null   int64  
     15  DISCOUNTED_PURCHASES    2240 non-null   int64  
     16  WEBSITE_PURCHASES       2240 non-null   int64  
     17  CATALOG_PURCHASES       2240 non-null   int64  
     18  STORE_PURCHASES         2240 non-null   int64  
     19  WEBSITE_VISITS          2240 non-null   int64  
     20  CAMPAIGN_3              2240 non-null   int64  
     21  CAMPAIGN_4              2240 non-null   int64  
     22  CAMPAIGN_5              2240 non-null   int64  
     23  CAMPAIGN_1              2240 non-null   int64  
     24  CAMPAIGN_2              2240 non-null   int64  
     25  COMPLAINED              2240 non-null   int64  
     26  Z_CostContact           2240 non-null   int64  
     27  Z_Revenue               2240 non-null   int64  
     28  ACCEPTED_LAST_CAMPAIGN  2240 non-null   int64  
    dtypes: float64(1), int64(25), object(3)
    memory usage: 507.6+ KB
    

Nice. Secondly, let's reorder them to match the data dictionary and drop Z_CostContact and Z_Revenue


```python
# Define the order of columns as per the data dictionary
ordered_columns = [
    'ID', 'BIRTH_YEAR', 'EDUCATION', 'MARITAL_STATUS', 'INCOME',
    'CHILDREN_AT_HOME', 'TEENS_AT_HOME', 'START_DATE', 'DAYS_SINCE_PURCHASE',
    'COMPLAINED', 'WINE', 'FRUIT', 'MEAT', 'FISH', 'SWEETS','GOLD',
    'DISCOUNTED_PURCHASES', 'CAMPAIGN_1', 'CAMPAIGN_2', 'CAMPAIGN_3',
    'CAMPAIGN_4', 'CAMPAIGN_5', 'ACCEPTED_LAST_CAMPAIGN', 'WEBSITE_PURCHASES',
    'CATALOG_PURCHASES', 'STORE_PURCHASES', 'WEBSITE_VISITS'
]

# Reorder the DataFrame columns
df = df[ordered_columns]
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>BIRTH_YEAR</th>
      <th>EDUCATION</th>
      <th>MARITAL_STATUS</th>
      <th>INCOME</th>
      <th>CHILDREN_AT_HOME</th>
      <th>TEENS_AT_HOME</th>
      <th>START_DATE</th>
      <th>DAYS_SINCE_PURCHASE</th>
      <th>COMPLAINED</th>
      <th>...</th>
      <th>CAMPAIGN_1</th>
      <th>CAMPAIGN_2</th>
      <th>CAMPAIGN_3</th>
      <th>CAMPAIGN_4</th>
      <th>CAMPAIGN_5</th>
      <th>ACCEPTED_LAST_CAMPAIGN</th>
      <th>WEBSITE_PURCHASES</th>
      <th>CATALOG_PURCHASES</th>
      <th>STORE_PURCHASES</th>
      <th>WEBSITE_VISITS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>04-09-2012</td>
      <td>58</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>10</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>08-03-2014</td>
      <td>38</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>21-08-2013</td>
      <td>26</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>10</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>10-02-2014</td>
      <td>26</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>19-01-2014</td>
      <td>94</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>10870</td>
      <td>1967</td>
      <td>Graduation</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>13-06-2013</td>
      <td>46</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>4001</td>
      <td>1946</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>10-06-2014</td>
      <td>56</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>2</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>7270</td>
      <td>1981</td>
      <td>Graduation</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>25-01-2014</td>
      <td>91</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
      <td>13</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>8235</td>
      <td>1956</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>24-01-2014</td>
      <td>8</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>5</td>
      <td>10</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>9405</td>
      <td>1954</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>15-10-2012</td>
      <td>40</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
<p>2240 rows × 27 columns</p>
</div>



It's coming along. Let's address the row count issue by dropping rows that contain nulls.


```python
original_row_count = len(df)
df = df.dropna()
print(f'{original_row_count-len(df)} rows have been dropped. The row count was {original_row_count} now is {len(df)}')
df.info()
```

    24 rows have been dropped. The row count was 2240 now is 2216
    <class 'pandas.core.frame.DataFrame'>
    Index: 2216 entries, 0 to 2239
    Data columns (total 27 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   ID                      2216 non-null   int64  
     1   BIRTH_YEAR              2216 non-null   int64  
     2   EDUCATION               2216 non-null   object 
     3   MARITAL_STATUS          2216 non-null   object 
     4   INCOME                  2216 non-null   float64
     5   CHILDREN_AT_HOME        2216 non-null   int64  
     6   TEENS_AT_HOME           2216 non-null   int64  
     7   START_DATE              2216 non-null   object 
     8   DAYS_SINCE_PURCHASE     2216 non-null   int64  
     9   COMPLAINED              2216 non-null   int64  
     10  WINE                    2216 non-null   int64  
     11  FRUIT                   2216 non-null   int64  
     12  MEAT                    2216 non-null   int64  
     13  FISH                    2216 non-null   int64  
     14  SWEETS                  2216 non-null   int64  
     15  GOLD                    2216 non-null   int64  
     16  DISCOUNTED_PURCHASES    2216 non-null   int64  
     17  CAMPAIGN_1              2216 non-null   int64  
     18  CAMPAIGN_2              2216 non-null   int64  
     19  CAMPAIGN_3              2216 non-null   int64  
     20  CAMPAIGN_4              2216 non-null   int64  
     21  CAMPAIGN_5              2216 non-null   int64  
     22  ACCEPTED_LAST_CAMPAIGN  2216 non-null   int64  
     23  WEBSITE_PURCHASES       2216 non-null   int64  
     24  CATALOG_PURCHASES       2216 non-null   int64  
     25  STORE_PURCHASES         2216 non-null   int64  
     26  WEBSITE_VISITS          2216 non-null   int64  
    dtypes: float64(1), int64(23), object(3)
    memory usage: 484.8+ KB
    

Great! Looks like the counts all align now. If we look at the Dtype column we can see that START_DATE is an object (string in this case) and should be a date type. Let's fix that. First we need to figure out what format it is in.


```python
df['START_DATE']
```




    0       04-09-2012
    1       08-03-2014
    2       21-08-2013
    3       10-02-2014
    4       19-01-2014
               ...    
    2235    13-06-2013
    2236    10-06-2014
    2237    25-01-2014
    2238    24-01-2014
    2239    15-10-2012
    Name: START_DATE, Length: 2216, dtype: object



Looks like it's probably going to be dd-mm-YYYY. Let's make sure that all the values match two digits, a dash, two digits, a dash, four digits (XX-XX-XXXX).


```python
# Regular expression pattern
pattern = r'^\d{2}-\d{2}-\d{4}$'

# Check the format of all values in the column
all_match_format = df['START_DATE'].str.match(pattern).all()

if all_match_format:
    print("All values are in format XX-XX-XXXX.")
else:
    print("Not all values are in format XX-XX-XXXX.")
```

    All values are in format XX-XX-XXXX.
    

Great, now let's figure out if the first set of digits is the day digits and the second set is the month digits.


```python
# Split the DataFrame
split_df = df['START_DATE'].str.split('-', expand=True)

# Analyze the range of values
first_component = split_df[0].astype(int)
second_component = split_df[1].astype(int)

print(f'First component max is: {first_component.max()}')
print(f'Second component max is: {second_component.max()}')
```

    First component max is: 31
    Second component max is: 12
    

Yeah, looks like the first component is the day and the second is the month. Let's change the column to a date type from format '%d-%m-%Y'


```python
# Update the start date column from a string object to a date type
df.loc[:, 'START_DATE'] = pd.to_datetime(df['START_DATE'], format='%d-%m-%Y')
df['START_DATE']
```




    0       2012-09-04 00:00:00
    1       2014-03-08 00:00:00
    2       2013-08-21 00:00:00
    3       2014-02-10 00:00:00
    4       2014-01-19 00:00:00
                   ...         
    2235    2013-06-13 00:00:00
    2236    2014-06-10 00:00:00
    2237    2014-01-25 00:00:00
    2238    2014-01-24 00:00:00
    2239    2012-10-15 00:00:00
    Name: START_DATE, Length: 2216, dtype: object



Success! Looks like some fields are 1 if they complained and 0 if not, but they are currently stored as an int and can be stored as a bool. Let's create a pragmatic function to update them instead of eyeballing it.


```python
def convert_int_to_bool(df):
    for col in df.columns:
        unique_values = df[col].unique()
        if set(unique_values) == {0, 1}:
            df.loc[:, col] = df[col].astype(bool)
    return df

df = convert_int_to_bool(df)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 2216 entries, 0 to 2239
    Data columns (total 27 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   ID                      2216 non-null   int64  
     1   BIRTH_YEAR              2216 non-null   int64  
     2   EDUCATION               2216 non-null   object 
     3   MARITAL_STATUS          2216 non-null   object 
     4   INCOME                  2216 non-null   float64
     5   CHILDREN_AT_HOME        2216 non-null   int64  
     6   TEENS_AT_HOME           2216 non-null   int64  
     7   START_DATE              2216 non-null   object 
     8   DAYS_SINCE_PURCHASE     2216 non-null   int64  
     9   COMPLAINED              2216 non-null   bool   
     10  WINE                    2216 non-null   int64  
     11  FRUIT                   2216 non-null   int64  
     12  MEAT                    2216 non-null   int64  
     13  FISH                    2216 non-null   int64  
     14  SWEETS                  2216 non-null   int64  
     15  GOLD                    2216 non-null   int64  
     16  DISCOUNTED_PURCHASES    2216 non-null   int64  
     17  CAMPAIGN_1              2216 non-null   bool   
     18  CAMPAIGN_2              2216 non-null   bool   
     19  CAMPAIGN_3              2216 non-null   bool   
     20  CAMPAIGN_4              2216 non-null   bool   
     21  CAMPAIGN_5              2216 non-null   bool   
     22  ACCEPTED_LAST_CAMPAIGN  2216 non-null   bool   
     23  WEBSITE_PURCHASES       2216 non-null   int64  
     24  CATALOG_PURCHASES       2216 non-null   int64  
     25  STORE_PURCHASES         2216 non-null   int64  
     26  WEBSITE_VISITS          2216 non-null   int64  
    dtypes: bool(7), float64(1), int64(16), object(3)
    memory usage: 378.7+ KB
    

Sweet! Looks like the COMPLAINED field and all the CAMPAIGN fields were updated. Now let's look at the unique values for all the column to see if there are any garbage entries


```python
for col in df.columns:
    print(df[col].value_counts())
```
    
Columns MARITAL_STATUS and EDUCATION have some funky entries. Now let's replace them with valid labels as we create new features.


```python
# If df is a slice from another DataFrame, create an explicit copy:
df = df.copy()

# Age of customer today
df["AGE"] = datetime.datetime.now().year - df["BIRTH_YEAR"]


# Total spending on various items
df["SPENT"] = df["WINE"] + df["FRUIT"] + df["MEAT"] + df["FISH"] + df["SWEETS"] + df["GOLD"]

# Feature indicating total children living in the household
df["CHILDREN"] = df["CHILDREN_AT_HOME"] + df["TEENS_AT_HOME"]

# Feature pertaining to parenthood
df.loc[:, "IS_PARENT"] = np.where(df["CHILDREN"] > 0, 1, 0)

# Calculate the difference in days between each date and the most recent date
df["CUSTOMER_TENURE"] = (pd.to_datetime(df["START_DATE"]).max() - pd.to_datetime(df["START_DATE"])).dt.days

# Segmenting education levels into three groups
df["EDUCATION"] = df["EDUCATION"].replace({
    "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
    "Graduation": "Graduate", "Master": "Postgraduate",
    "PhD": "Postgraduate"
})

# Deriving living situation by marital status "Alone"
df["LIVING_WITH"] = df["MARITAL_STATUS"].replace({
    "Married": "Partner", "Together": "Partner",
    "Absurd": "Alone", "Widow": "Alone",
    "YOLO": "Alone", "Divorced": "Alone",
    "Single": "Alone"
})

# Feature for total members in the household
df.loc[:, "FAMILY_SIZE"] = df["LIVING_WITH"].replace({"Alone": 1, "Partner": 2}) + df["CHILDREN"]

# Drop what we don't need
col_to_drop = ["ID", "BIRTH_YEAR", "MARITAL_STATUS", "START_DATE"]
df = df.drop(col_to_drop, axis=1)

df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INCOME</th>
      <th>CHILDREN_AT_HOME</th>
      <th>TEENS_AT_HOME</th>
      <th>DAYS_SINCE_PURCHASE</th>
      <th>WINE</th>
      <th>FRUIT</th>
      <th>MEAT</th>
      <th>FISH</th>
      <th>SWEETS</th>
      <th>GOLD</th>
      <th>...</th>
      <th>WEBSITE_PURCHASES</th>
      <th>CATALOG_PURCHASES</th>
      <th>STORE_PURCHASES</th>
      <th>WEBSITE_VISITS</th>
      <th>AGE</th>
      <th>SPENT</th>
      <th>CHILDREN</th>
      <th>IS_PARENT</th>
      <th>CUSTOMER_TENURE</th>
      <th>FAMILY_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>...</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
      <td>2216.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>52247.251354</td>
      <td>0.441787</td>
      <td>0.505415</td>
      <td>49.012635</td>
      <td>305.091606</td>
      <td>26.356047</td>
      <td>166.995939</td>
      <td>37.637635</td>
      <td>27.028881</td>
      <td>43.965253</td>
      <td>...</td>
      <td>4.085289</td>
      <td>2.671029</td>
      <td>5.800993</td>
      <td>5.319043</td>
      <td>55.179603</td>
      <td>607.075361</td>
      <td>0.947202</td>
      <td>0.714350</td>
      <td>353.521209</td>
      <td>2.592509</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25173.076661</td>
      <td>0.536896</td>
      <td>0.544181</td>
      <td>28.948352</td>
      <td>337.327920</td>
      <td>39.793917</td>
      <td>224.283273</td>
      <td>54.752082</td>
      <td>41.072046</td>
      <td>51.815414</td>
      <td>...</td>
      <td>2.740951</td>
      <td>2.926734</td>
      <td>3.250785</td>
      <td>2.425359</td>
      <td>11.985554</td>
      <td>602.900476</td>
      <td>0.749062</td>
      <td>0.451825</td>
      <td>202.434667</td>
      <td>0.905722</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35303.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>47.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>180.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51381.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>174.500000</td>
      <td>8.000000</td>
      <td>68.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>24.500000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>54.000000</td>
      <td>396.500000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>355.500000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68522.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>74.000000</td>
      <td>505.000000</td>
      <td>33.000000</td>
      <td>232.250000</td>
      <td>50.000000</td>
      <td>33.000000</td>
      <td>56.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>65.000000</td>
      <td>1048.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>529.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>666666.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>321.000000</td>
      <td>...</td>
      <td>27.000000</td>
      <td>28.000000</td>
      <td>13.000000</td>
      <td>20.000000</td>
      <td>131.000000</td>
      <td>2525.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>699.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>



Great! Let's look for outliers now. Looks like there may be outliners. The 75% income is around 68k and the max is 666k. We should plot the outliers.


```python
import matplotlib.colors as mcolors

# Setting up color preferences for the plot
sns.set(rc={'axes.facecolor': '#EDEFF0', 'figure.facecolor': '#EDEFF0'})

# Defining a color palette with a list of hex color codes
color_palette_heatmap = ['#4B0082', '#7B68EE', '#ADD8E6', '#90EE90', '#FFFFE0', '#FFA07A', '#FF4500']

# Creating a color map from the modern color palette
color_map = mcolors.LinearSegmentedColormap.from_list("color_palette_heatmap", color_palette_heatmap)

# Defining the list of features to be plotted
features_to_plot = ['INCOME', 'AGE', 'IS_PARENT']
print("Relative Plot of Selected Features: Data Subset Analysis")

# Creating the pairplot with the selected features
plt.figure()
sns.pairplot(df[features_to_plot], hue='IS_PARENT', palette=color_palette_heatmap)

# Displaying the plot
plt.show()

```

    Relative Plot of Selected Features: Data Subset Analysis
    

    C:\Users\A2667879\git\virtual_enviornments\demographic_segmentation\lib\site-packages\seaborn\axisgrid.py:1513: UserWarning: The palette list has more values (7) than needed (2), which may not be intended.
      func(x=vector, **plot_kwargs)
    C:\Users\A2667879\git\virtual_enviornments\demographic_segmentation\lib\site-packages\seaborn\axisgrid.py:1513: UserWarning: The palette list has more values (7) than needed (2), which may not be intended.
      func(x=vector, **plot_kwargs)
    C:\Users\A2667879\git\virtual_enviornments\demographic_segmentation\lib\site-packages\seaborn\axisgrid.py:1615: UserWarning: The palette list has more values (7) than needed (2), which may not be intended.
      func(x=x, y=y, **kwargs)
    C:\Users\A2667879\git\virtual_enviornments\demographic_segmentation\lib\site-packages\seaborn\axisgrid.py:1615: UserWarning: The palette list has more values (7) than needed (2), which may not be intended.
      func(x=x, y=y, **kwargs)
    


    <Figure size 800x550 with 0 Axes>



    
![png](marketing_segmentation/output_27_3.png)
    


As you can see, there are some outliers stranded way out there for INCOME and AGE. I'll remove those.


```python
df[df['AGE']>100]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDUCATION</th>
      <th>INCOME</th>
      <th>CHILDREN_AT_HOME</th>
      <th>TEENS_AT_HOME</th>
      <th>DAYS_SINCE_PURCHASE</th>
      <th>COMPLAINED</th>
      <th>WINE</th>
      <th>FRUIT</th>
      <th>MEAT</th>
      <th>FISH</th>
      <th>...</th>
      <th>CATALOG_PURCHASES</th>
      <th>STORE_PURCHASES</th>
      <th>WEBSITE_VISITS</th>
      <th>AGE</th>
      <th>SPENT</th>
      <th>CHILDREN</th>
      <th>IS_PARENT</th>
      <th>CUSTOMER_TENURE</th>
      <th>LIVING_WITH</th>
      <th>FAMILY_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>192</th>
      <td>Undergraduate</td>
      <td>36640.0</td>
      <td>1</td>
      <td>0</td>
      <td>99</td>
      <td>True</td>
      <td>15</td>
      <td>6</td>
      <td>8</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>124</td>
      <td>65</td>
      <td>1</td>
      <td>1</td>
      <td>276</td>
      <td>Alone</td>
      <td>2</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Undergraduate</td>
      <td>60182.0</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>False</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>131</td>
      <td>22</td>
      <td>1</td>
      <td>1</td>
      <td>43</td>
      <td>Alone</td>
      <td>2</td>
    </tr>
    <tr>
      <th>339</th>
      <td>Postgraduate</td>
      <td>83532.0</td>
      <td>0</td>
      <td>0</td>
      <td>36</td>
      <td>False</td>
      <td>755</td>
      <td>144</td>
      <td>562</td>
      <td>104</td>
      <td>...</td>
      <td>6</td>
      <td>4</td>
      <td>1</td>
      <td>125</td>
      <td>1853</td>
      <td>0</td>
      <td>0</td>
      <td>276</td>
      <td>Partner</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>



Looks like people with an age over 100 have outlived the longest lived human in history - must be old data.


```python
df[df['INCOME']>160000]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>EDUCATION</th>
      <th>INCOME</th>
      <th>CHILDREN_AT_HOME</th>
      <th>TEENS_AT_HOME</th>
      <th>DAYS_SINCE_PURCHASE</th>
      <th>COMPLAINED</th>
      <th>WINE</th>
      <th>FRUIT</th>
      <th>MEAT</th>
      <th>FISH</th>
      <th>...</th>
      <th>CATALOG_PURCHASES</th>
      <th>STORE_PURCHASES</th>
      <th>WEBSITE_VISITS</th>
      <th>AGE</th>
      <th>SPENT</th>
      <th>CHILDREN</th>
      <th>IS_PARENT</th>
      <th>CUSTOMER_TENURE</th>
      <th>LIVING_WITH</th>
      <th>FAMILY_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>617</th>
      <td>Postgraduate</td>
      <td>162397.0</td>
      <td>1</td>
      <td>1</td>
      <td>31</td>
      <td>False</td>
      <td>85</td>
      <td>1</td>
      <td>16</td>
      <td>2</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>48</td>
      <td>107</td>
      <td>2</td>
      <td>1</td>
      <td>391</td>
      <td>Partner</td>
      <td>4</td>
    </tr>
    <tr>
      <th>687</th>
      <td>Postgraduate</td>
      <td>160803.0</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>False</td>
      <td>55</td>
      <td>16</td>
      <td>1622</td>
      <td>17</td>
      <td>...</td>
      <td>28</td>
      <td>1</td>
      <td>0</td>
      <td>42</td>
      <td>1717</td>
      <td>0</td>
      <td>0</td>
      <td>694</td>
      <td>Partner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2233</th>
      <td>Graduate</td>
      <td>666666.0</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
      <td>False</td>
      <td>9</td>
      <td>14</td>
      <td>18</td>
      <td>8</td>
      <td>...</td>
      <td>1</td>
      <td>3</td>
      <td>6</td>
      <td>47</td>
      <td>62</td>
      <td>1</td>
      <td>1</td>
      <td>392</td>
      <td>Partner</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>



Same for income, there's a single outlier that skews the mean. I'll cap the income at 163k, just above the second-highest income.


```python
# Set maxim
df = df[(df["AGE"]<=100)] # Max age set to 100
df = df[(df["INCOME"]<163000)] # Max income set to 175
df.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INCOME</th>
      <th>CHILDREN_AT_HOME</th>
      <th>TEENS_AT_HOME</th>
      <th>DAYS_SINCE_PURCHASE</th>
      <th>WINE</th>
      <th>FRUIT</th>
      <th>MEAT</th>
      <th>FISH</th>
      <th>SWEETS</th>
      <th>GOLD</th>
      <th>...</th>
      <th>WEBSITE_PURCHASES</th>
      <th>CATALOG_PURCHASES</th>
      <th>STORE_PURCHASES</th>
      <th>WEBSITE_VISITS</th>
      <th>AGE</th>
      <th>SPENT</th>
      <th>CHILDREN</th>
      <th>IS_PARENT</th>
      <th>CUSTOMER_TENURE</th>
      <th>FAMILY_SIZE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>...</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
      <td>2212.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51958.810579</td>
      <td>0.441682</td>
      <td>0.505877</td>
      <td>49.019439</td>
      <td>305.287523</td>
      <td>26.329566</td>
      <td>167.029837</td>
      <td>37.648734</td>
      <td>27.046564</td>
      <td>43.925859</td>
      <td>...</td>
      <td>4.088156</td>
      <td>2.672242</td>
      <td>5.806510</td>
      <td>5.321429</td>
      <td>55.086347</td>
      <td>607.268083</td>
      <td>0.947559</td>
      <td>0.714286</td>
      <td>353.714286</td>
      <td>2.593128</td>
    </tr>
    <tr>
      <th>std</th>
      <td>21527.278844</td>
      <td>0.536955</td>
      <td>0.544253</td>
      <td>28.943121</td>
      <td>337.322940</td>
      <td>39.744052</td>
      <td>224.254493</td>
      <td>54.772033</td>
      <td>41.090991</td>
      <td>51.706981</td>
      <td>...</td>
      <td>2.742187</td>
      <td>2.927542</td>
      <td>3.250939</td>
      <td>2.425597</td>
      <td>11.701599</td>
      <td>602.513364</td>
      <td>0.749466</td>
      <td>0.451856</td>
      <td>202.494886</td>
      <td>0.906236</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1730.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>28.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>35233.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>24.000000</td>
      <td>24.000000</td>
      <td>2.000000</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>9.000000</td>
      <td>...</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>47.000000</td>
      <td>69.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>180.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51371.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>175.500000</td>
      <td>8.000000</td>
      <td>68.000000</td>
      <td>12.000000</td>
      <td>8.000000</td>
      <td>24.500000</td>
      <td>...</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>5.000000</td>
      <td>6.000000</td>
      <td>54.000000</td>
      <td>397.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>356.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>68487.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>74.000000</td>
      <td>505.000000</td>
      <td>33.000000</td>
      <td>232.250000</td>
      <td>50.000000</td>
      <td>33.000000</td>
      <td>56.000000</td>
      <td>...</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>7.000000</td>
      <td>65.000000</td>
      <td>1048.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>529.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>162397.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>99.000000</td>
      <td>1493.000000</td>
      <td>199.000000</td>
      <td>1725.000000</td>
      <td>259.000000</td>
      <td>262.000000</td>
      <td>321.000000</td>
      <td>...</td>
      <td>27.000000</td>
      <td>28.000000</td>
      <td>13.000000</td>
      <td>20.000000</td>
      <td>84.000000</td>
      <td>2525.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>699.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 21 columns</p>
</div>



Ah, much better. Time to plot correlation.


```python
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(correlation,annot=True, cmap=color_palette_heatmap, center=0)
```




    <Axes: >




    
![png](marketing_segmentation/output_35_1.png)
    


## Data Preprocessing

In this segment, my focus will be on preparing the data for clustering tasks.

The data preprocessing involves several key steps:

- Applying label encoding to the categorical features.
- Utilizing the standard scaler for feature scaling.
- Generating a subset dataframe for reducing dimensionality.

**Label Encoding Categorical Features**
Label encoding transforms categorical data into numbers, making it compatible with algorithms that require numerical input.



```python
# Identify columns with categorical data
categorical_columns = [column for column in df.columns if df[column].dtype == 'object']
print("List of categorical variables:", categorical_columns)

# Apply label encoding to categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

print("All features have been converted to numerical format")
```

    List of categorical variables: ['EDUCATION', 'LIVING_WITH']
    All features have been converted to numerical format
    

**Feature Scaling**
- Feature scaling adjusts all data features to a similar scale, so that no single feature dominates just because it has larger values.

```python
# Duplicate the original dataframe for modifications
dataset_copy = df.copy()

# Remove specific columns that are not needed
columns_to_remove = ['CAMPAIGN_1', 'CAMPAIGN_2', 'CAMPAIGN_3', 'CAMPAIGN_4', 'CAMPAIGN_5', 'COMPLAINED', 'ACCEPTED_LAST_CAMPAIGN']
dataset_copy.drop(columns=columns_to_remove, inplace=True)

# Apply standard scaling to the dataset
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
scaled_dataset = pd.DataFrame(standard_scaler.fit_transform(dataset_copy), columns=dataset_copy.columns)
print("Features have been successfully scaled")


```

    Features have been successfully scaled
    

## PCA Dimensionality Reduction
Dimensionality Reduction simplifies the data by merging multiple features into a smaller number of important ones, making it easier to analyze.

I'll simplify the data by using a technique called PCA to reduce the number of features, focusing on the most important ones, and then plot the streamlined data, aiming for a 3D representation.


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Applying PCA to reduce dimensions to 3
principal_component_analyzer = PCA(n_components=3)
reduced_data = pd.DataFrame(principal_component_analyzer.fit_transform(scaled_dataset), columns=["DIMENSION_1", "DIMENSION_2", "DIMENSION_3"])
print(reduced_data.describe().transpose())

# Visualizing the reduced dimensions in 3D
figure = plt.figure(figsize=(10, 8))
axis = figure.add_subplot(111, projection='3d')
axis.scatter(reduced_data['DIMENSION_1'], reduced_data['DIMENSION_2'], reduced_data['DIMENSION_3'], color='blue')
axis.set_title("3D Projection of Principal Components")
plt.show()

```

                  count          mean       std       min       25%       50%  \
    DIMENSION_1  2212.0 -1.220643e-16  2.878602 -5.978092 -2.539468 -0.781593   
    DIMENSION_2  2212.0 -6.424437e-17  1.709469 -4.194901 -1.324292 -0.173801   
    DIMENSION_3  2212.0  8.833601e-18  1.231651 -3.629258 -0.849351 -0.048129   
    
                      75%       max  
    DIMENSION_1  2.386383  7.452936  
    DIMENSION_2  1.232639  6.168705  
    DIMENSION_3  0.857177  6.785164  
    


    
![png](marketing_segmentation/output_42_1.png)
    


## Clustering

Now that we've simplified our data to three dimensions aspects, we need to group similar data points together. We use a technique called "Agglomerative clustering," where we start by considering each data point as its own group and gradually combine the closest groups until we have the desired number of clusters.

- Find Optimal Cluster Count: We figure out how many groups we should have using the Elbow Method, which helps us decide on the ideal number of clusters.

- Agglomerative Clustering: With Agglomerative clustering, we put similar data points into the same groups based on their similarities.

- Visualizing Clustered Data: To understand the results, we create a scatter plot that shows the groups, making it easier to see patterns and relationships in the data.

**Find Optimal Cluster Count**


```python
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# Determine the optimal number of clusters using the Elbow Method
kmeans_estimator = KMeans()
elbow_visualizer = KElbowVisualizer(kmeans_estimator, k=10)

# Fit the visualizer to the reduced data and display the plot
elbow_visualizer.fit(reduced_data)
elbow_visualizer.show()
```


    
![png](marketing_segmentation/output_45_0.png)
    





    <Axes: title={'center': 'Distortion Score Elbow for KMeans Clustering'}, xlabel='k', ylabel='distortion score'>



The optimal number of clusters is 4. We will plug that into our AgglomerativeClustering Model

**Agglomerative Clustering**


```python
from sklearn.cluster import AgglomerativeClustering

# Apply Agglomerative Clustering
agglomerative_clustering = AgglomerativeClustering(n_clusters=4)
clusters = agglomerative_clustering.fit_predict(reduced_data)
reduced_data['CLUSTERS'] = clusters
df['CLUSTERS'] = clusters

# Create a 3D scatter plot to visualize clusters
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(reduced_data['DIMENSION_1'], reduced_data['DIMENSION_2'], reduced_data['DIMENSION_3'], s=40, c=reduced_data["CLUSTERS"], marker='o')
ax.set_title("The Plot Of The Clusters")
plt.show()

```


    
![png](marketing_segmentation/output_48_0.png)
    


## Evaluation Models
**Exploratory Data Analysis of Clusters**

In this analysis, we're sorting data into groups based on similarities, but we don't have a clear way to check if our sorting is perfect. So, we'll look closely at these groups, explore the data inside them, and try to understand what makes each group unique. This helps us find interesting patterns and insights in the data.

**Distribution of Clusters**


```python
import seaborn as sns

# Visualize the distribution of clusters
sns.countplot(data=df, x='CLUSTERS')
plt.title("Cluster Distribution")
plt.show()

```


    
![png](marketing_segmentation/output_51_0.png)
    


The distribution plot shows that the clusters appear to have a relatively balanced.

**Cluster Patterning**


```python
# Create a scatter plot to show cluster patterns based on income and spending
sns.scatterplot(data=df, x='SPENT', y='INCOME', hue='CLUSTERS')
plt.title("Cluster Patterns: Income vs Spent")
plt.show()
```


    
![png](marketing_segmentation/output_54_0.png)
    


The pattern plot that shows there are four types of spending/income groups:
- Budget: low income & low spending
- Impulse: low income & high spending
- Average: average income & average spending
- Wealthy: high income & high spending


To get all clusters properly labeled, let's use quantiles to define cutoffs for income and spending.
We will use 50th percentile (median) for the cutoff between 'Budget' and 'Impulse'/'Average' and the same for 'Wealthy'. Additionally, we will introduce a new cutoff at 75th percentile for income and 25th percentile for spending to differentiate between 'Impulse' and 'Average'.


```python
# Calculate the 50th and 75th percentiles for income and the 25th percentile for spending
income_50th_percentile = df['INCOME'].quantile(0.5)
income_75th_percentile = df['INCOME'].quantile(0.75)
spent_25th_percentile = df['SPENT'].quantile(0.25)

# Redefine the labeling conditions using these new percentiles
def label_cluster_quantiles(row):
    if row['INCOME'] <= income_50th_percentile and row['SPENT'] <= spent_25th_percentile:
        return "Budget"
    elif row['INCOME'] > income_75th_percentile and row['SPENT'] > spent_25th_percentile:
        return "Wealthy"
    elif row['INCOME'] <= income_50th_percentile and row['SPENT'] > spent_25th_percentile:
        return "Impulse"
    else:  # This covers cases where income is between the 50th and 75th percentile or spending is at or below the 25th percentile
        return "Average"

# Apply the new labeling conditions to the test DataFrame
df['CLUSTERS'] = df.apply(label_cluster_quantiles, axis=1)
```

**Product Spending Distribution**
Next, let's examine how the different grocery products are distributed within each cluster.


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
pl = sns.stripplot(x=df["CLUSTERS"], y=df["SPENT"], color="#CBEDDD", alpha=0.5, jitter=True)
pl = sns.boxenplot(x=df["CLUSTERS"], y=df["SPENT"])
plt.show()
```


    
![png](marketing_segmentation/output_58_0.png)
    


Based on the graph provided, it is evident that our largest customer group is Wealthy, with Average being a close second. We can delve into the spending habits of each cluster to devise targeted marketing strategies. Let us next explore how did our campaigns do in the past.


```python
# Plotting count of total campaign accepted.
df["TOTAL_CAMPAIGN"] = df["CAMPAIGN_1"] + df["CAMPAIGN_2"] + df["CAMPAIGN_3"] + df["CAMPAIGN_4"] + df["CAMPAIGN_5"]

plt.figure()
plot = sns.countplot(x=df["TOTAL_CAMPAIGN"], hue=df["CLUSTERS"])
plot.set_title("Campaigns Accepted")
plot.set_xlabel("Was accepted")

# Show the plot
plt.show()
```


    
![png](marketing_segmentation/output_60_0.png)
    


So far, the campaigns have failed more than succeeded. We need to focus and carefully plan targeted campaigns to each cluster. We know the campaigns need work, but what about the deals?


```python
# Plotting the number of deals purchased
plt.figure()
pl=sns.boxenplot(y=df["DISCOUNTED_PURCHASES"],x=df["CLUSTERS"])
pl.set_title("Discounts Purchased")
plt.show()
```


    
![png](marketing_segmentation/output_62_0.png)
    


Unlike campaigns, the deals we rolled out actually did pretty well. They performed the best among the Average group and the Impulse shoppers. However, it seems our high-value Wealthy customers weren't all that interested in the deals. Surprisingly, the Budget crowd didn't show much enthusiasm either.

## Profiling

Let's identify customers in each shopping group and determine our favorites and those needing more attention from the marketing team based on their shopping habits.

In order to market effectively we should plot the new features to profile each cluster.


```python
def map_age_to_generation(age):
    if age >= 11 and age <= 26:
        return 'GenZ'
    elif age >= 27 and age <= 42:
        return 'Millennials'
    elif age >= 43 and age <= 58:
        return 'GenX'
    elif age >= 59 and age <= 77:
        return 'Baby Boomers'
    elif age >= 78 and age <= 95:
        return 'Silent Generation'
    else:
        return 'Greatest Generation'

# Apply the function to create the 'Generation' column
df['GENERATION'] = df['AGE'].apply(map_age_to_generation)

# Aggregate data
generation_counts = df['GENERATION'].value_counts()

# Plot
generation_counts.plot(kind='bar')
plt.xlabel('Generation')
plt.ylabel('Count')
plt.title('Generation Counts')
plt.show()
```


    
![png](marketing_segmentation/output_65_0.png)
    


Most customers are older. Looks like Millennial's need to be targeted.


```python
traits = ['AGE', 'FAMILY_SIZE', 'CHILDREN_AT_HOME', 'TEENS_AT_HOME', 'INCOME']

for trait in traits:
    sns.jointplot(data=df, x=trait, y='SPENT', hue='CLUSTERS', kind='kde')
    plt.show()
```
![png](marketing_segmentation/output_67_4.png)

    
![png](marketing_segmentation/output_67_0.png)
    



    
![png](marketing_segmentation/output_67_1.png)
    



    
![png](marketing_segmentation/output_67_2.png)
    



    
![png](marketing_segmentation/output_67_3.png)
    


# Findings

**Budget Cluster**:
- Identified as the least valuable cluster with a limited number of individuals.
- Represents a diverse range of generations and family compositions.
- Primarily consists of older individuals.
- Characterized by lower income levels and minimal spending habits.
- Typically includes parents with families of 1-5 members, likely having children and possibly a teenager at home.
- Marketing campaigns have shown limited effectiveness, suggesting a need for more targeted approaches.
- Surprisingly, this group tends to use fewer discounts than expected.

**Impulse Cluster**:
- Recognized as a moderately valuable cluster with a moderate number of members.
- Comprises a broad spectrum of generations.
- Generally characterized by low income but higher spending tendencies.
- Commonly includes parents with a family size of 2-4, typically with one child at home.
- Marketing campaigns have had limited success with this cluster.
- This group tends to utilize a reasonable number of discounts.

**Average Cluster**:
- Considered a highly valuable cluster with a substantial number of members.
- Predominantly consists of individuals from Generation X and Baby Boomers.
- Characterized by average income and spending levels.
- Typically includes parents with a family size of 2-3, potentially with a teenager still living at home.
- Marketing campaigns have been moderately effective with this group.
- This cluster is known for making good use of discounts.

**Wealthy Cluster**:
- Rated as the most valuable cluster with a significant number of members and highest spending.
- Primarily composed of Generation X and Baby Boomers.
- Characterized by high income and significant spending habits.
- Most members are couples, often without children, or if they are parents, they may have just one teenager at home.
- Marketing campaigns have achieved moderate success with this group.
- Members of this cluster tend to use discounts less frequently.

# Marketing Recommendations:

**Budget Cluster Marketing Ideas**:
- Family-Focused Deals: Since many in this cluster are parents with up to 5 people in the family, including children and possibly a teen, we should highlight offers that cater to family needs and budgets. This could mean bundling products or offering family-sized deals.
- Parent-Friendly Rewards: Our loyalty program could include special perks for parents, like discounts on kids' items or family activities. This could encourage more frequent visits from larger families.
- Customized Communication for Families: Tailoring our emails or mailers with family-oriented content and offers can make a big difference. It's all about showing them we understand their family dynamics and needs.

**Impulse Cluster Marketing Ideas**:
- Promotions for Parents with Kids: This group typically consists of parents with a smaller family size, usually with one child at home. Our promotions can focus on products that appeal to both parents and children, encouraging spontaneous purchases.
- Family-Friendly Online Content: Our digital content should be engaging and appealing to both parents and kids. Think about interactive social media campaigns or family-oriented contests.
- Streamlined Shopping for Busy Parents: We should aim for a shopping experience that's quick and hassle-free, especially for parents who might be shopping with a child.

**Average Cluster Marketing Ideas**:
- Balanced Family Offers: With most in this cluster being parents, often with a teenager at home, our marketing should balance offers that appeal to both adults and teens. This might include tech gadgets, clothing, or entertainment options.
- Diverse Ad Strategies for Families: Since they have diverse interests spanning generations, a combination of digital and traditional marketing can effectively reach these family-oriented consumers.
- Testimonials from Families: Sharing stories or reviews from families, particularly those with teenagers, can resonate more with this group and build a deeper connection.

**Wealthy Cluster Marketing Ideas**:
- Exclusive Offers for Sophisticated Families: This group, often consisting of couples or parents with a teenager, looks for premium experiences. We can offer exclusive family packages or high-end products that cater to a sophisticated lifestyle.
- Subtle, High-End Family Marketing: Our advertising should be understated yet luxurious, appealing to their desire for exclusivity and quality. We can think about sponsoring upscale family events or partnering with luxury family-oriented brands.
- VIP Service for Parents and Teens: Providing top-notch customer service, especially catering to the needs of parents with teens, can enhance their shopping experience and reinforce their loyalty to our brand.

