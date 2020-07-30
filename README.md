# MarketBasketAnalysis
The project uses MLxtend in python using retail data

It is used by retailers to understand the customer purchasing pattern. It looks for combinations of item that occurs in the purchase transaction. Asociation analysis is an unsupervised learning tool that helps to identify hidden pattern. This method require very little preprocessing and feature engineering. 
Association rule, uses few keywords like antecedent, consequent, support, confidence and lift.
The rules are commonly written as {cocacola}->{chips}, means the customer who buy coca cola has a very high probability to buy chips. 
In this {cococola} is known as antecedent and {chips} is consequent.Support is the frequency of the occurrence of the rule, confidence the measure of reliability and lift the ratio of the observed support to that expected if the two rules were independent.

### Python packages
install mlxtend

```python
pip install mlxtend

```

```python
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

```
The data used in the project is online retail II from UCI repository. It has purchase transaction data of 42 countries.

```python
data=pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx')
data.head(5)

```

```python
data['Country'].unique()
```

### Data preprocessing

The description in the data has space which  need to removed. Then there are invoices without number, in this project these invoices are dropped. Also credit transaction invoices are removed from the set.

```python
data['Description'] = data['Description'].str.strip()# remove space in description
data.dropna(axis=0, subset=['Invoice'], inplace=True)#drop invoice without a number
data['Invoice'] = data['Invoice'].astype('str')
data = data[~data['Invoice'].str.contains('C')]#Remove credit transaction invoices
```

once the data cleaning is completed, the data is converted as 1 transaction per row and create basket. The first one I create is for USA

```python
basket = (data [data ['Country'] =="USA"]
          .groupby(['Invoice', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('Invoice'))
```
Then all the positive values are converted to 1 and others are set as 0

```python
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_USA = basket.applymap(encode_units)
```
I am not including the postage charge in this project, however that can also be considered in the future 
```python
basket_USA.drop('POSTAGE', inplace=True, axis=1)
```
### Model Generation

```python
frequent_itemsets = apriori(basket_USA, min_support=0.07, use_colnames=True)
```
Now a rule is generated 

```python
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.head()
```
Then I have filtered the data which has 7 lift and 80% confidence

```python
rules[ (rules['lift'] >= 7) &
       (rules['confidence'] >= 0.8) ]
```



