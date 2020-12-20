# pandemicdatahack-track3
### Команда: Работа не волк. Работа - это work
Kaggle: https://www.kaggle.com/c/pandemicdatahack/overview

Дополнительные данные: [ссылка](https://drive.google.com/drive/folders/1EUdZ6O1XtNw4Y109QbPUfF2gefmp89Ml?usp=sharing)

Котировки на нефть, газ, валюты взяты [отсюда](https://ru.investing.com/commodities/natural-gas-historical-data); данные по COVID-19 взяты с [Yandex DataLens](https://datalens.yandex/7o7is1q6ikh23?tab=ov3&utm_source=cbfooter); данные по ВВП, безработице и инфляции собраны с разных сайтов. [Эмбеддинги fasttext](http://docs.deeppavlov.ai/en/master/features/pretrained_vectors.html)
![](img.jpg)


### Описание решения:
1. Настроили 5-Fold валидацию со стратификацией по году публикации резюме, потому что в train и test у них схожие распределения. Локальная валидация коррелировала с public leaderboard’ом.
2. Для теста усредняли предсказания моделей, обученных на разных фолдах.
3. 1 в таргете является выбросом, который портит модель. Поэтому сначала делали классификацию (CatBoost Classifier): выброс (salary < 300) или нет. На не выбросах обучали CatBoost Regressor на логарифмированном таргете, оптимизируя MSE. Выбрали именно CatBoost из-за встроенной обработка категориальных признаков и поддержки GPU.
4. На тестовых данных запускали обе модели, итоговое предсказание = (значение регрессии) * (вероятность от классификации).
5. По исходным данным был большой препроцессинг, его вы можете найти в файле preprocess.py
6. Использовали дополнительные данные (с 2015 по 2020 года): ВВП России по годам в рублях и долларах; ежедневные котировки евро, доллара, нефти, газа, золота; количество больных COVID-19 по дням и дням-регионам.
7. Для текстовых полей из данных по образовании брался усредненный вектор fasttext (100-dim) по словам в предложении. Текст предобрабатывался с помощью удаления html-тэгов и лемматизация с pymorphy

### Обзор данных:
1. [eda 1](notebooks/eda.ipynb)

1. [eda 2](notebooks/eda_my.ipynb)


### Предобработка данных:

1. [Основной скрипт предобработки](notebooks/preprocess.py)

2. [Добавление внешних данных](notebooks/make_dataset.ipynb)

3. [Очистка текстовых данных в employements.csv](notebooks/clean_text_in_employements.ipynb)

4. [Fasttext embeddings для текстовых полей](src/get_embeddings.py)


### Обучение и предсказание моделей:

1. [Обучение catboost моделей](notebooks/catboost-baseline.ipynb)

2. [Обучение моделей Random Forest и Linear Reagression](notebooks/rf_and_linreg_baseline.ipynb)

3. [Обучение моделей LightGBM](notebooks/optuna.ipynb) и код для обучения в папке [src/](src/)

4. [Усреднение различных предсказаний](notebooks/submit-averaging.ipynb)


### Сабмиты
1. [Папка с сабмитами](submits/)
