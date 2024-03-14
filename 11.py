import pandas as pd

# Чтение CSV файла в DataFrame
df = pd.read_csv('cluster.csv')

# Преобразование столбцов с датами в тип datetime
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='mixed')
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='mixed')

# Разбиение DataFrame на кластеры по годам и кластерам
clusters = df.groupby(['year', 'cluster'])

# Проход по каждому кластеру
for (year, cluster), data in clusters:
    print(f"Год: {year}")
    print(f"Кластер: {cluster}")

    # Выявление характеристик, проявляющих закономерности

    # Пример: среднее значение количества пассажиров
    passenger_count_mean = data['passenger_count'].mean()
    print(f"Среднее количество пассажиров: {passenger_count_mean}")

    # Пример: суммарное расстояние поездок
    trip_distance_sum = data['trip_distance'].sum()
    print(f"Суммарное расстояние поездок: {trip_distance_sum}")

    print()