# House Prices: Exploratory Data Analysis

Exploratory data analysis of the [Ames Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (1,460 residential properties sold in Ames, Iowa, 2006–2010), with a machine learning model to predict sale prices.

## Goal

Identify which factors have the greatest influence on house pricing — and test specific hypotheses about how those factors interact.

## What's inside

- **Data cleaning** — handling missing values in `LotFrontage` and `GarageType`
- **Feature engineering** — `HouseAgeSold`, `PricePerSqFt`
- **Visualization** — distribution plots, correlation heatmap, regression plots, comparative analysis across neighborhoods and garage types
- **Hypothesis testing** — two-condition hypotheses validated with t-tests (e.g. *do houses with average heating but larger living area cost more than houses with excellent heating but smaller area?*)
- **Machine learning** — CatBoostRegressor trained on 7 features, achieving **9.2% mean absolute error** (~$16,653) on house price prediction

## Key findings

- `GenQuality` and `LivArea` are the strongest predictors of price (correlation 0.79 and 0.71); `LotArea` barely matters (0.26) — buyers value living space over plot size
- The 2008 financial crisis is visible in the data: average prices dropped from $186k (2007) to $177k (2008) and never fully recovered within the dataset period
- New houses with an attached garage have a significantly higher price per sq ft than new houses with a built-in garage — a non-obvious result, since built-in garages are generally more convenient

## Tools

Python · pandas · matplotlib · seaborn · plotly · scipy · CatBoost
