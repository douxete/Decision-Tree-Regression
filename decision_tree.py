import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# read the csv fie
melbourne_file_path = "./dataset/melb_data.csv"

melbourne_data = pd.read_csv(melbourne_file_path)

# print the columns
print(melbourne_data.columns)

# dropping data that has missing values
melbourne_data = melbourne_data.dropna(axis=0)

# select the prediction data target
y = melbourne_data.Price

# chose the features
# features will be used to make predictions
mebourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[mebourne_features]

# describe the features
print(X.describe())
# print few of the top rows
print(X.head())

# using sklearn to define the model, and specify a number for random_state to get the same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model
melbourne_model.fit(X,y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions")
print(melbourne_model.predict(X.head()))

# validate the model using MAE
predicted_home_prices = melbourne_model.predict(X)
print("Mean Absolute Error Without Splitting :")
print(mean_absolute_error(y, predicted_home_prices))

# split the data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define the model
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(train_X, train_y)

print("Mean Absolute Error With Splitting :")
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# underfitting and overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

print("Mean Absolute Error With Max Leaf Nodes :")
print(get_mae(5, train_X, val_X, train_y, val_y))

print("Mean Absolute Error With Looping Max Leaf Nodes :")
for max_leaf_nodes in [5, 10, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))