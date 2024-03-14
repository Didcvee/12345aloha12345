from flask import Flask, render_template, request
from sklearn.cluster import KMeans
import pandas as pd
from joblib import load
from flask import Flask, render_template, request
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Загрузка модели
model = load('model.joblib')


app = Flask(__name__)

from flask import send_file




@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/aloha', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/clusterize', methods=['POST'])
def clusterize():
    # Получаем загруженный файл из запроса
    file = request.files['file']

    # Читаем csv-файл в DataFrame
    df = pd.read_csv(file)

    # Получаем число кластеров из запроса
    num_clusters = int(request.form.get('clusters'))

    # Создаем модель KMeans
    kmeans = KMeans(n_clusters=num_clusters)

    # Кластеризуем данные
    clusters = kmeans.fit_predict(df)

    # Добавляем столбец 'Cluster' в DataFrame
    df['Cluster'] = clusters

    # Сохраняем откластеризованные данные в новый csv-файл
    output_filename = 'output.csv'
    df.to_csv(output_filename, index=False)

    return render_template('result.html', filename=output_filename)


# Обработка формы и генерация прогноза
@app.route('/predict', methods=['POST'])
def predict():
    # Получение данных из формы
    cluster = int(request.form['cluster'])
    period = int(request.form['period'])

    # Загрузка CSV-файла
    csv_file = request.files['csv_file']
    df = pd.read_csv(csv_file)

    # Подготовка данных для предсказания
    X = df[['passenger_count', 'trip_distance', 'ratecodeid', 'pulocationid', 'dolocationid', 'payment_type',
            'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
            'congestion_surcharge', 'Month', 'year', 'cluster']]
    X = X[X['cluster'] == cluster]
    X['year'] = X['year'] + period
    X = X.drop('cluster', axis=1)

    # Прогноз на новых данных
    predictions = model.predict(X)

    # График прогноза
    plt.figure()
    sns.lineplot(x=X['year'], y=predictions)
    plt.xlabel('Year')
    plt.ylabel('Total Amount')
    plt.title('Forecast for Cluster {}'.format(cluster))

    # Преобразование графика в изображение
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Возвращаем сгенерированное изображение
    return render_template('result1.html', plot_url=plot_url)


@app.route('/download/<filename>')
def download(filename):
    return send_file(filename, as_attachment=True)



if __name__ == '__main__':
    app.run()
