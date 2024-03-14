import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных
data = pd.read_csv('cluster.csv')

# Выделение нужных колонок для расчета среднего значения
columns_to_average = ['total_amount']

# Расчет среднего значения по году и кластеру для каждой колонки
averages = data.groupby(['year', 'cluster'])[columns_to_average].mean().reset_index()

# Создание столбчатой диаграммы
# Создание столбчатой диаграммы для каждого кластера
clusters = data['cluster'].unique()
for cluster in clusters:
    cluster_averages = averages[averages['cluster'] == cluster]
    plt.bar(cluster_averages['year'], cluster_averages['total_amount'], label=f'Кластер {cluster}')

    plt.xlabel('Год')
    plt.ylabel('Среднее количество пассажиров')
    plt.title('Столбчатая диаграмма среднего количества пассажиров по году и кластеру')
    plt.xticks(cluster_averages['year'].unique())
    plt.legend()
    plt.show()