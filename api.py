"""
REST API (FastAPI) для распознавания цветков на изображениях.

Запуск:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ml_utils import (
    ModelNotReadyError,
    format_prediction,
    load_model_bundle,
    predict_proba,
    preprocess_image_bytes,
    SUPPORTED_EXT,
)

model = None
class_names: list[str] = []
metrics: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, class_names, metrics
    model, class_names, metrics = load_model_bundle()
    yield


app = FastAPI(
    title="Flower Recognition API",
    description=(
        "API для классификации цветков на изображениях. "
        "Модель: MobileNetV2 (Keras Applications), датасет: tf_flowers."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ClassPrediction(BaseModel):
    class_name: str
    class_index: int
    probability: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float = Field(ge=0.0, le=1.0)
    top_predictions: list[ClassPrediction]


class BatchPredictionItem(BaseModel):
    filename: str
    predicted_class: str | None = None
    predicted_index: int | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    top_predictions: list[ClassPrediction] | None = None
    error: str | None = None


class BatchPredictionResponse(BaseModel):
    count: int
    results: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    classes: list[str]
    val_accuracy: float | None = None
    dataset: str | None = None


class RootResponse(BaseModel):
    message: str
    docs: str
    health: str
    predict: str
    classes: str


@app.get("/", response_model=RootResponse, tags=["info"])
def root():
    return RootResponse(
        message="Flower Recognition API — распознавание цветков на изображениях",
        docs="/docs",
        health="/health",
        predict="POST /predict (multipart/form-data, поле file)",
        classes="/classes",
    )


@app.get("/health", response_model=HealthResponse, tags=["info"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        num_classes=len(class_names),
        classes=class_names,
        val_accuracy=metrics.get("val_accuracy"),
        dataset=metrics.get("dataset"),
    )


@app.get("/classes", response_model=list[str], tags=["info"])
def get_classes():
    return class_names


def _validate_image_upload(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Имя файла не указано.")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_EXT:
        supported = ", ".join(sorted(SUPPORTED_EXT))
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла. Поддерживаются: {supported}",
        )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_image(
    file: Annotated[UploadFile, File(description="Изображение цветка (jpg, png, webp и др.)")],
):
    _validate_image_upload(file)

    try:
        content = await file.read()
        batch = preprocess_image_bytes(content)
        proba = predict_proba(model, batch)[0]
        result = format_prediction(proba, class_names, top_k=3)
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Не удалось обработать изображение: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
async def predict_batch(
    files: Annotated[
        list[UploadFile],
        File(description="Один или несколько файлов изображений"),
    ],
):
    if not files:
        raise HTTPException(status_code=400, detail="Нужно загрузить хотя бы одно изображение.")

    results: list[BatchPredictionItem] = []
    valid_batches: list[np.ndarray] = []
    valid_meta: list[str] = []

    for upload in files:
        try:
            _validate_image_upload(upload)
            content = await upload.read()
            valid_batches.append(preprocess_image_bytes(content)[0])
            valid_meta.append(upload.filename or "unknown")
        except HTTPException as exc:
            results.append(
                BatchPredictionItem(
                    filename=upload.filename or "unknown",
                    error=exc.detail if isinstance(exc.detail, str) else str(exc.detail),
                )
            )

    if valid_batches:
        batch = np.stack(valid_batches)
        proba = predict_proba(model, batch)
        for filename, row in zip(valid_meta, proba):
            item = format_prediction(row, class_names, top_k=3)
            results.append(
                BatchPredictionItem(
                    filename=filename,
                    predicted_class=item["predicted_class"],
                    predicted_index=item["predicted_index"],
                    confidence=item["confidence"],
                    top_predictions=item["top_predictions"],
                )
            )

    return BatchPredictionResponse(count=len(results), results=results)


def create_app() -> FastAPI:
    """Фабрика для тестов или альтернативного запуска."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
