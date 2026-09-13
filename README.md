Вариант Распознавание дорожных знаков
БВТ2253 Доронин Егор Евгеньевич

Установка и запуск, обучение, запуск и демо приложения
```
pip install -r requirements.txt
python train.py --config configs/densenet121.yaml
python evaluate.py --config configs/densenet121.yaml --ckpt models/densenet121_best.pth
streamlit run app.py
```
Структура
```
configs/ — YAML-конфиги обучения
dataset.py — загрузка и аугментация GTSRB
models.py — 5 архитектур
train.py, evaluate.py — обучение и оценка
generate_report.py, plot_curves.py — отчёты
app.py — Streamlit-приложение
```

Датасет
```
GTSRB (German Traffic Sign Recognition Benchmark):
43 класса дорожных знаков
39 209 train + 12 630 test
```
