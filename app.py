import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
import urllib
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.stats import ttest_ind
from sklearn.model_selection import cross_val_score

def process_data():
    data = pd.read_csv('FertilizersProduct.csv', encoding='ISO-8859-1')
    print("Original data size:", data.shape)
    data = data[data['Item'] == "Ammonia, anhydrous"]
    data = data[(data['Year'] >= 2007) & (data['Year'] <= 2017)]
    data = data.pivot_table(values='Value', index=['Area', 'Year', 'Element'], columns='Unit', aggfunc='sum')
    import_data = data[data.index.get_level_values('Element').str.contains('Import')].infer_objects()
    export_data = data[data.index.get_level_values('Element').str.contains('Export')].infer_objects()
    import_data.index = import_data.index.set_names(['Country', 'Year', 'Element'])
    export_data.index = export_data.index.set_names(['Country', 'Year', 'Element'])
    top_import_countries = import_data.groupby(level=0).sum().nlargest(20, 'tonnes')
    top_export_countries = export_data.groupby(level=0).sum().nlargest(20, 'tonnes')
    print("Top 20 import countries:\n", top_import_countries)
    print("Top 20 export countries:\n", top_export_countries)
    return top_import_countries, top_export_countries, import_data, export_data
top_import_countries, top_export_countries, import_data, export_data = process_data()


def generate_heatmap(data, title):
    correlation_matrix = data.corr()
    correlation_matrix = correlation_matrix[::-1]
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', cbar_kws={'label': 'Correlation Coefficient'})
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=14)
    plt.show()

top_10_import_countries = import_data['1000 US$'].groupby(level=0).sum().nlargest(10).index
top_10_export_countries = export_data['1000 US$'].groupby(level=0).sum().nlargest(10).index
import_data_top = import_data[import_data.index.get_level_values('Country').isin(top_10_import_countries) & import_data.index.get_level_values('Country').isin(export_data.index.get_level_values('Country'))]
export_data_top = export_data[export_data.index.get_level_values('Country').isin(top_10_export_countries)]

import_data_pivot = import_data_top['1000 US$'].unstack(level=0)
import_correlation_matrix = import_data_pivot.corr()
export_data_pivot = export_data_top['1000 US$'].unstack(level=0)
export_correlation_matrix = export_data_pivot.corr()

print("Correlation Matrix for Top 10 Import Countries:\n", import_correlation_matrix)
generate_heatmap(import_correlation_matrix, 'Top 10 Import Countries')

print("Correlation Matrix for Top 10 Export Countries:\n", export_correlation_matrix)
generate_heatmap(export_correlation_matrix, 'Top 10 Export Countries')
plt.savefig('correlation_matrix2.png') 


top_20_import_countries = import_data['1000 US$'].groupby(level=0).sum().nlargest(20).index
import_data_top_20 = import_data[import_data.index.get_level_values('Country').isin(top_20_import_countries)]
import_data_sum = import_data_top_20['1000 US$'].groupby(level=0).sum()
import_data_sum.sort_values(ascending=False).plot(kind='bar', color='magenta')
plt.title('Bar Plot of Import Values for Top 20 Import Countries')
plt.xlabel('Country')
plt.ylabel('Import Value (1000 US$)')
plt.show()

top_20_export_countries = export_data['1000 US$'].groupby(level=0).sum().nlargest(20).index
top_20_export_countries = top_20_export_countries.drop('Trinidad and Tobago')
export_data_top_20 = export_data[export_data.index.get_level_values('Country').isin(top_20_export_countries)]
export_data_sum = export_data_top_20['1000 US$'].groupby(level=0).sum()
export_data_sum.sort_values(ascending=False).plot(kind='bar', color='blue')
plt.title('Bar Plot of Export Values for Top 20 Export Countries')
plt.xlabel('Country')
plt.ylabel('Export Value (1000 US$)')
plt.show()


top_5_data_export = export_data[export_data.index.get_level_values('Country').isin(top_20_export_countries[:5])]
top_5_mean_export = top_5_data_export['1000 US$'].mean()
top_5_std_export = top_5_data_export['1000 US$'].std()
print(f'Top 5 Export Countries - Mean: {top_5_mean_export}, Standard Deviation: {top_5_std_export}')

next_15_data_export = export_data[export_data.index.get_level_values('Country').isin(top_20_export_countries[5:])]
next_15_mean_export = next_15_data_export['1000 US$'].mean()
next_15_std_export = next_15_data_export['1000 US$'].std()
print(f'Next 15 Export Countries - Mean: {next_15_mean_export}, Standard Deviation: {next_15_std_export}')

from scipy.stats import ttest_ind
top_5_data_export = top_5_data_export['1000 US$'].dropna()
next_15_data_export = next_15_data_export['1000 US$'].dropna()
t_stat_export, p_val_export = ttest_ind(top_5_data_export, next_15_data_export)
print(f'T-test for Export Data - T Statistic: {t_stat_export}, P Value: {p_val_export}')


top_5_data_import = import_data[import_data.index.get_level_values('Country').isin(top_20_import_countries[:5])]
top_5_mean_import = top_5_data_import['1000 US$'].mean()
top_5_std_import = top_5_data_import['1000 US$'].std()
print(f'Top 5 Import Countries - Mean: {top_5_mean_import}, Standard Deviation: {top_5_std_import}')
next_15_data_import = import_data[import_data.index.get_level_values('Country').isin(top_20_import_countries[5:])]
next_15_mean_import = next_15_data_import['1000 US$'].mean()
next_15_std_import = next_15_data_import['1000 US$'].std()
print(f'Next 15 Import Countries - Mean: {next_15_mean_import}, Standard Deviation: {next_15_std_import}')


top_5_data_import = top_5_data_import['1000 US$'].dropna()
next_15_data_import = next_15_data_import['1000 US$'].dropna()
t_stat_import, p_val_import = ttest_ind(top_5_data_import, next_15_data_import)
print(f'T-test for Import Data - T Statistic: {t_stat_import}, P Value: {p_val_import}')



top_20_countries = export_data['1000 US$'].groupby(level=0).sum().nlargest(20).index
top_20_countries = top_20_countries.drop('Trinidad and Tobago')
models = {}
next_10_years_data = {}
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
for country in top_20_countries:
    country_data = export_data.xs(country, level='Country').copy()
    for i in range(1, 4):
        country_data[f'FertilizerExport_lag{i}'] = country_data['1000 US$'].shift(i)
    country_data = country_data.fillna(0)
    if country_data.empty:
        continue
    X = country_data.drop(['1000 US$'], axis=1)
    y = country_data['1000 US$']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    models[country] = best_rf
    next_10_years = pd.DataFrame(data=0, index=np.arange(10), columns=X_train.columns)
    next_10_years_data[country] = next_10_years

for country, model in models.items():
    next_10_years = next_10_years_data[country].copy()  
    print(f'Making predictions for {country}.')
    predictions = []
    for year in range(10):
        input_data = next_10_years.loc[year][model.feature_names_in_]
        input_data = pd.DataFrame([input_data], columns=model.feature_names_in_)
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        next_10_years.at[year, 'Predicted'] = prediction[0]  
        if year < 9: 
            for i in range(3, 1, -1):
                next_10_years[f'FertilizerExport_lag{i}'] = next_10_years[f'FertilizerExport_lag{i-1}']
            next_10_years.rename(columns={'FertilizerExport_Lag1': 'FertilizerExport_lag1'}, inplace=True)  
    print(f'Predictions for {country}: {predictions}')



scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Average cross-validation score: {scores.mean()}')
import matplotlib.pyplot as plt


top_5_countries = export_data['1000 US$'].groupby(level=0).sum().nlargest(5).index.tolist()
predicted_exports = {country: models[country].predict(next_10_years_data[country]).sum() for country in models.keys()}
predicted_countries = sorted(predicted_exports, key=predicted_exports.get, reverse=True)[:5]
countries = list(set(top_5_countries + predicted_countries))
years = list(range(2007, 2028))
plt.figure(figsize=(10, 5))

for country in countries:
    if country == 'Trinidad and Tobago':
        continue
    actual_data = export_data.xs(country, level='Country')['1000 US$']
    predicted_data = models[country].predict(next_10_years_data[country])
    data = pd.concat([actual_data, pd.Series(predicted_data, index=range(2018, 2028))])
    years = list(range(2007, 2007 + len(data)))
    plt.plot(years, data, marker='o', linestyle='', label=country)
plt.axvline(x=2018, color='r', linestyle='--')
plt.xlim(2007, 2027)
plt.title('Fertilizer Export Predictions 2007-2027')
plt.xlabel('Year')
plt.ylabel('Fertilizer Export')
plt.legend()
plt.show()


top_20_countries = import_data['1000 US$'].groupby(level=0).sum().nlargest(20).index
models = {}
next_10_years_data = {}
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

for country in top_20_countries:
    country_data = import_data.xs(country, level='Country').copy()
    for i in range(1, 4):  # Change this to create fewer lagged features
        country_data[f'FertilizerExport_lag{i}'] = country_data['1000 US$'].shift(i)
    country_data = country_data.fillna(0)
    if country_data.empty:
        print(f'No data for {country}. Skipping...')
        continue
    X = country_data.drop(['1000 US$'], axis=1)
    y = country_data['1000 US$']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    models[country] = best_rf
    next_10_years = pd.DataFrame(data=0, index=np.arange(10), columns=X_train.columns)
    next_10_years_data[country] = next_10_years


for country, model in models.items():
    next_10_years = next_10_years_data[country].copy()  
    print(f'Making predictions for {country}.')
    predictions = []

    for year in range(10):
        input_data = next_10_years.loc[year][model.feature_names_in_]
        input_data = pd.DataFrame([input_data], columns=model.feature_names_in_)
        prediction = model.predict(input_data)
        predictions.append(prediction[0])
        next_10_years.at[year, 'Predicted'] = prediction[0]  

        if year < 9: 
            for i in range(3, 1, -1):
                next_10_years[f'FertilizerExport_lag{i}'] = next_10_years[f'FertilizerExport_lag{i-1}']
            next_10_years.rename(columns={'FertilizerExport_Lag1': 'FertilizerExport_lag1'}, inplace=True)  
    print(f'Predictions for {country}: {predictions}')


scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-validation scores: {scores}')
print(f'Average cross-validation score: {scores.mean()}')


top_5_countries = import_data['1000 US$'].groupby(level=0).sum().nlargest(5).index.tolist()
predicted_import = {country: models[country].predict(next_10_years_data[country]).sum() for country in models.keys()}
predicted_countries = sorted(predicted_exports, key=predicted_exports.get, reverse=True)[:5]
countries = list(set(top_5_countries + predicted_countries))
years = list(range(2007, 2028))
plt.figure(figsize=(10, 5))

for country in countries:
    if country not in models:
        print(f"No model for {country}. Skipping...")
        continue
    actual_data = import_data.xs(country, level='Country')['1000 US$']
    predicted_data = models[country].predict(next_10_years_data[country])
    data = pd.concat([actual_data, pd.Series(predicted_data, index=range(2018, 2028))])
    years = list(range(2007, 2007 + len(data)))
    plt.plot(years, data, marker='o', linestyle='', label=country)
plt.axvline(x=2018, color='r', linestyle='--')

plt.xlim(2007, 2027)
plt.title('Fertilizer Import Predictions 2007-2027')
plt.xlabel('Year')
plt.ylabel('Fertilizer Export')
plt.legend()
plt.show()