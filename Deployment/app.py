from flask import Flask, request, render_template
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

app = Flask(__name__, template_folder='template')

# Load the dataset
data = pd.read_csv('clean_data.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Extract items from form
    items = request.form.getlist('items')
    items = [item.strip() for item in items]

    # Check if all items are present in the dataset
    missing_items = [item for item in items if item not in data.columns]

    if missing_items:
        return render_template('not_found.html', missing_items=missing_items)

    # Filter basket based on selected items
    filtered_basket = data[items]

    # Generate association rules
    frequent_itemsets = apriori(filtered_basket, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

    # Filter rules to keep only relevant ones
    filtered_rules = rules[
        (rules['antecedents'].apply(lambda x: set(items).issubset(x))) &  
        (rules['consequents'].apply(lambda x: len(x) >= 1)) &
        (rules['lift'] > 1.5) &
        (rules['confidence'] > 0.7)
    ]

    # Get recommendations
    recommendations = filtered_rules['consequents'].apply(lambda x: list(x)[0]).tolist()

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
