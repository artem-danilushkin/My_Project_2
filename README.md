# My_Project_2 - API на FastAPI
API предоставляет HTTP‑интерфейс к модели: можно отправить изображение и получить предсказание в формате JSON.
### Запуск FastAPI‑сервера
Из каталога `flowers/` (в активированном виртуальном окружении):

```powershell
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

После запуска:

- базовый URL: `http://127.0.0.1:8000`
- документация Swagger UI: `http://127.0.0.1:8000/docs`
- документация ReDoc: `http://127.0.0.1:8000/redoc`

Остановить сервер: **Ctrl + C** в окне терминала.

### Основные эндпоинты

- **`GET /`** — краткая информация о сервисе и ссылках (`/docs`, `/health`, `/predict`).
- **`GET /health`** — статус модели:
  - загружена ли модель,
  - сколько классов,
  - accuracy на валидации,
  - название датасета.
- **`GET /classes`** — список всех классов (названия цветков).
- **`POST /predict`** — предсказание для одного изображения (multipart/form‑data, поле `file`).
- **`POST /predict/batch`** — предсказание для нескольких изображений сразу (поле `files`).

#### Предсказание для одного изображения (через Swagger)

1. Откройте `http://127.0.0.1:8000/docs`.
2. Найдите `POST /predict` → нажмите **Try it out**.
3. В поле `file` выберите изображение (`.jpg`, `.png`, `.webp` и др.).
4. Нажмите **Execute** и посмотрите JSON‑ответ.

#### Предсказание для одного изображения (через curl)

```bash
curl -X POST "http://127.0.0.1:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@C:\Users\Artem\My_project_Cursor\flowers\images\rose.jpg"
```

Пример ответа:

```json
{
  "predicted_class": "roses",
  "predicted_index": 4,
  "confidence": 0.982341,
  "top_predictions": [
    {"class_name": "roses", "class_index": 4, "probability": 0.982341},
    {"class_name": "tulips", "class_index": 2, "probability": 0.012345},
    {"class_name": "daisy", "class_index": 1, "probability": 0.003210}
  ]
}
```
