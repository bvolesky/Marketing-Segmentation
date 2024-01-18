# Why?
Customer Personality Analysis is a detailed analysis of a company’s ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.
<br/>
Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. For example, instead of spending money to market a new product to every customer in the company’s database, a company can analyze which customer segment is most likely to buy the product and then market the product only on that particular segment.
<br/>
## Data Dictionary
### People
   - **ID**: Customer's unique identifier
   - **BIRTH_YEAR**: Customer's birth year
   - **EDUCATION**: Customer's education level
   - **MARITAL_STATUS**: Customer's marital status
   - **INCOME**: Customer's yearly household income
   - **CHILDREN_AT_HOME**: Number of children in customer's household
   - **TEENS_AT_HOME**: Number of teenagers in customer's household
   - **START_DATE**: Date of customer's enrollment with the company
   - **DAYS_SINCE_PURCHASE**: Number of days since customer's last purchase
   - **COMPLAINED**: 1 if the customer complained in the last 2 years, 0 otherwise

### Products
   - **WINE**: Amount spent on wine in last 2 years
   - **FRUIT**: Amount spent on fruits in last 2 years
   - **MEAT**: Amount spent on meat in last 2 years
   - **FISH**: Amount spent on fish in last 2 years
   - **SWEETS**: Amount spent on sweets in last 2 years

### Promotion
   - **DISCOUNTED_PURCHASES**: Number of purchases made with a discount
   - **CAMPAIGN_1**: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
   - **CAMPAIGN_2**: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
   - **CAMPAIGN_3**: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
   - **CAMPAIGN_4**: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
   - **CAMPAIGN_5**: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
   - **CAMPAIGN_6**: 1 if customer accepted the offer in the 6th campaign, 0 otherwise
   - **RESPONSE**: 1 if customer accepted the offer in the last campaign, 0 otherwise

### Place
   - **WEBSITE_PURCHASES**: Number of purchases made through the company’s website
   - **CATALOG_PURCHASES**: Number of purchases made using a catalog
   - **STORE_PURCHASES**: Number of purchases made directly in stores
   - **WEBSITE_VISITS**: Number of visits to company’s website in the last month

## Goal
Need to perform clustering to summarize customer segments.