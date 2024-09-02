import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
weather = pd.read_csv("3788103.csv", index_col = "DATE")

# Find what percentage of rows in each column are null
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]

# Find columns/variables that have less than 5% missing data and make new data set
valid_columns = weather.columns[null_pct < 0.05]
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()

# Makes sure each column does not have any missing values
weather = weather.ffill()

#Convert object "DATE" to datetime data type
weather.index = pd.to_datetime(weather.index)

# Create target column which predicts tomorrow's max temperature, use ffill for last value
weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()

rr = Ridge(alpha=.1)

def backtest(weather, model, predictors, start=3650, step=90):
    all_predictions = []

    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]

        model.fit(train[predictors], train["target"])

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index = test.index)
        combined = pd.concat([test["target"], preds], axis = 1)

        combined.columns = ["actual", "prediction"]

        combined["diff"] = (combined["prediction"] - combined["actual"]).abs()

        all_predictions.append(combined)
    return pd.concat(all_predictions)

def pct_diff(old, new):
    return (new-old) / old

def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"

    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather

rolling_horizons = [3, 14]

for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

weather = weather.iloc[14:,:]
weather = weather.fillna(0)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys= False).apply(expand_mean)

predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictions = backtest(weather, rr, predictors)
x = mean_absolute_error(predictions["actual"], predictions["prediction"])
y = predictions.sort_values("diff", ascending = False)

