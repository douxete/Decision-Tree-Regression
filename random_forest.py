import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# read the csv fie
melbourne_file_path = "./dataset/melb_data.csv"

melbourne_data = pd.read_csv(melbourne_file_path)

# dropping data that has missing values
melbourne_data = melbourne_data.dropna(axis=0)

# select the prediction data target
y = melbourne_data.Price

# chose the features
# features will be used to make predictions
mebourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[mebourne_features]

# split the data into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))