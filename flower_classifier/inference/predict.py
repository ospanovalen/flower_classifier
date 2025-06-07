import json
from pathlib import Path
from typing import Dict, Union

import click
import torch
from PIL import Image

from flower_classifier.data.dataset import get_default_transforms
from flower_classifier.models.flower_model import FlowerClassifier


class FlowerPredictor:
    """Single image flower prediction."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize predictor.

        Args:
            model_path: Path to model checkpoint
            device: Device to use for inference
        """
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.transforms = get_default_transforms(size=224)
        self.class_names = {
            0: "daisy",
            1: "tulips",
            2: "sunflowers",
            3: "dandelion",
            4: "roses",
        }

    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, model_path: str) -> FlowerClassifier:
        """Load model from checkpoint."""
        model = FlowerClassifier.load_from_checkpoint(model_path)
        model.eval()
        model.to(self.device)
        return model

    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict flower class for single image.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with prediction results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)

        # Get prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()

        # Convert to human-readable format
        predicted_class = self.class_names[predicted_class_idx]
        all_probabilities = {
            self.class_names[i]: probabilities[0, i].item()
            for i in range(len(self.class_names))
        }

        return {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities,
        }


@click.command()
@click.option("--image-path", required=True, help="Path to input image")
@click.option("--model-path", required=True, help="Path to model checkpoint")
@click.option("--device", default="auto", help="Device to use (auto/cpu/cuda)")
@click.option("--output-file", help="Optional: save results to JSON file")
def main(image_path: str, model_path: str, device: str, output_file: str):
    """Predict flower class for a single image."""
    predictor = FlowerPredictor(model_path, device)
    result = predictor.predict(image_path)

    print(json.dumps(result, indent=2))

    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
