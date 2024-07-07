
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Load the dataset
data = pd.read_csv('transactions.csv')

# Prepare the data for the Apriori algorithm
transactions = data.groupby(['TransactionID'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm
frequent_itemsets_apriori = apriori(df, min_support=0.2, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.5)

# Print Apriori results
print("Apriori Frequent Itemsets:")
print(frequent_itemsets_apriori)
print("\nApriori Rules:")
print(rules_apriori)

# Implementing the Eclat algorithm
def get_support(df, itemsets):
    n_transactions = df.shape[0]
    support = df[itemsets].all(axis=1).sum() / n_transactions
    return support

def eclat(prefix, items, min_support):
    while items:
        item = items.pop()
        new_prefix = prefix + [item]
        support = get_support(df, new_prefix)
        if support >= min_support:
            frequent_itemsets.append((new_prefix, support))
            suffix = [x for x in items if x > item]
            eclat(new_prefix, suffix, min_support)

# Generate frequent itemsets using Eclat
min_support = 0.2
frequent_itemsets = []
items = list(df.columns)
eclat([], items, min_support)

# Print Eclat results
print("\nEclat Frequent Itemsets:")
for itemset, support in frequent_itemsets:
    print(f"Itemset: {itemset}, Support: {support:.2f}")
