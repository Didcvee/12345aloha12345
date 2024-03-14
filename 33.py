import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv("cluster.csv")

# Преобразуем столбцы с датами в тип данных datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='mixed')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='mixed')

# Разделим датасет на признаки (X) и целевую переменную (y)
X = df[['passenger_count', 'trip_distance', 'ratecodeid', 'pulocationid', 'dolocationid', 'payment_type', 'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'congestion_surcharge', 'Month', 'year', 'cluster']]
y = df['total_amount']

# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

import matplotlib.pyplot as plt
import numpy as np

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot(np.arange(0, 1000), np.arange(0, 1000), color='red')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('График фактических значений vs. предсказанных значений')
plt.show()


from joblib import dump

# Сохранение модели
dump(model, 'model.joblib')

from joblib import load

# Загрузка модели
model = load('model.joblib')

# Использование модели для предсказания
new_data = pd.read_csv("new_data.csv")  # Подготовьте данные, на основе которых хотите получить прогноз
predictions = model.predict(new_data)
