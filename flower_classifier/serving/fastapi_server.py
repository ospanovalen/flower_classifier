"""FastAPI inference server for flower classification."""

import io
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from ..data.dataset import get_default_transforms
from ..models.flower_model import FlowerClassifier

logger = logging.getLogger(__name__)


class PredictionResponse(BaseModel):
    """Prediction response model."""

    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""

    predictions: List[PredictionResponse]


class FlowerInferenceServer:
    """FastAPI server for flower classification inference."""

    CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize inference server.

        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = None
        self.transforms = get_default_transforms(size=224)
        self._load_model()

    def _load_model(self):
        """Load model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")

        try:
            # Load model
            self.model = FlowerClassifier.load_from_checkpoint(
                self.model_path, map_location=self.device
            )
            self.model.eval()
            self.model.to(self.device)

            logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def predict_image(self, image: Image.Image) -> PredictionResponse:
        """Predict class for single image.

        Args:
            image: PIL Image

        Returns:
            Prediction response
        """
        try:
            # Preprocess image
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Transform image
            input_tensor = self.transforms(image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Make prediction
            with torch.no_grad():
                logits = self.model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)[0]

            # Convert to response
            probs_dict = {
                class_name: float(prob)
                for class_name, prob in zip(self.CLASSES, probabilities)
            }

            predicted_idx = torch.argmax(probabilities)
            predicted_class = self.CLASSES[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            return PredictionResponse(
                predicted_class=predicted_class,
                confidence=confidence,
                all_probabilities=probs_dict,
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


# Global inference server instance
inference_server = None


def create_app() -> FastAPI:
    """Create FastAPI app.

    Returns:
        FastAPI app instance
    """
    global inference_server

    # Get parameters from environment variables
    model_path = os.environ.get("FLOWER_MODEL_PATH")
    device = os.environ.get("FLOWER_DEVICE", "auto")

    if not model_path:
        raise ValueError("FLOWER_MODEL_PATH environment variable must be set")

    app = FastAPI(
        title="Flower Classification API",
        description="REST API for flower classification using deep learning",
        version="1.0.0",
    )

    # Initialize inference server
    inference_server = FlowerInferenceServer(model_path, device)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Flower Classification API",
            "version": "1.0.0",
            "model_path": str(inference_server.model_path),
            "device": str(inference_server.device),
            "classes": inference_server.CLASSES,
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)):
        """Predict flower class for uploaded image.

        Args:
            file: Uploaded image file

        Returns:
            Prediction response
        """
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        try:
            # Read and process image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Make prediction
            result = inference_server.predict_image(image)

            return result

        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    async def predict_batch(files: List[UploadFile] = File(...)):
        """Predict flower classes for multiple uploaded images.

        Args:
            files: List of uploaded image files

        Returns:
            Batch prediction response
        """
        if len(files) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

        predictions = []

        for file in files:
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400, detail=f"File {file.filename} must be an image"
                )

            try:
                # Read and process image
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))

                # Make prediction
                result = inference_server.predict_image(image)
                predictions.append(result)

            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Error processing {file.filename}: {e}"
                )

        return BatchPredictionResponse(predictions=predictions)

    return app


if __name__ == "__main__":
    import uvicorn

    # Example usage
    model_path = "models/best_model.ckpt"
    app = create_app()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
