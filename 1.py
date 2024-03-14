import pandas as pd

# Загрузка CSV файлов
df1 = pd.read_csv('result.csv')
df2 = pd.read_csv('new_result.csv')

# Объединение всех трех файлов
df = pd.concat([df1, df2])
df['year'] = pd.to_datetime(df['tpep_pickup_datetime'], format='mixed').dt.year
# Получение пропорционального количества записей для каждого года
n_samples = 1000
sample_per_year = n_samples // len(df['year'].unique())

# Фильтрация данных до 1000 записей с пропорциональным количеством для каждого года
filtered_df = df.groupby('year').head(sample_per_year)

# Сохранение результата в новый CSV файл
filtered_df.to_csv('slitok.csv', index=False)