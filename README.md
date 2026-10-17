# Forte Vision — baseline для кассовых видео

Это локальный MVP для поиска визуальных отклонений от normal-поведения (`ANOMALY` / `NORMAL`). Anomaly не является юридическим названием нарушения и не определяет его тип.

## Идея

CLIP — предобученная модель зрения. Мы не дообучаем её на 42 роликах: это было бы ненадёжно. Вместо этого CLIP превращает каждый кадр в **embedding** — числовой вектор, описывающий визуальное содержание кадра.

Ролик делится на **temporal windows**: по умолчанию окна длиной 10 секунд с шагом 5 секунд. Для каждого окна берутся 2 кадра в секунду. Для их embedding считаются mean, std и max; объединённый вектор передаётся в `LogisticRegression`.

Для anomaly baseline CLIP остаётся frozen. Модель нормы fit-ится только на окнах normal training videos; violation timestamps не участвуют в fit и используются лишь для финальной evaluation/localization.

Важно: split делается **по роликам**, никогда не по окнам. Иначе похожие кадры одного ролика попадут и в train, и в test, а метрика будет нечестной.

Итоговая вероятность ролика — максимум вероятностей его окон. Если она не меньше порога (по умолчанию 0.5), verdict — `VIOLATION`; подозрительный timestamp — центр окна с максимальной вероятностью. Порог исследуется по precision/recall/F1: для кейса особенно важен recall нарушений.

## Установка и запуск

```bash
cd Chiaki
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/inspect_dataset.py
python scripts/audit_labels.py
python scripts/train_anomaly.py --rebuild-features
python scripts/evaluate_anomaly.py
python scripts/debug_anomaly.py
uvicorn app.main:app --reload
```

Откройте http://127.0.0.1:8000. Первый запуск OpenCLIP может скачать веса модели. Данные должны лежать в `dataset/`, включая `open_labels.xlsx`.

`train_anomaly.py` сохраняет fixed test IDs из прежнего diagnostic split, отделяет normal validation videos от normal train videos и выбирает threshold как 95-й percentile normal validation video scores. Violation test labels никогда не участвуют в fit или подборе threshold.

`debug_predictions.py` создаёт `outputs/heldout_windows.csv` (по убыванию вероятности) и кадры TP/FP/FN/TN в `outputs/evidence_frames/`; это только диагностические материалы, не данные обучения.

Новый feature cache использует только fixed inference grid. Он содержит absolute CLIP statistics и temporal delta/magnitude statistics. Сравниваются kNN distance, IsolationForest и OneClassSVM, а также absolute/temporal/combined представления; все их fit-ы normal-only.

## Ограничения baseline

Датасет очень мал, timestamp размечает только один момент, а простая агрегация теряет порядок движений. Поэтому результат — измеримая отправная точка, а не готовая production-модель. Следующие эксперименты: другие CLIP-энкодеры, подбор окон/порога, более строгая CV и temporal-модель.
