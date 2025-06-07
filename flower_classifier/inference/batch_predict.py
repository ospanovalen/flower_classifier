import json
from pathlib import Path
from typing import Dict, List

import click
from tqdm import tqdm

from flower_classifier.inference.predict import FlowerPredictor


class BatchFlowerPredictor:
    """Batch flower prediction for multiple images."""

    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize batch predictor.

        Args:
            model_path: Path to model checkpoint
            device: Device to use for inference
        """
        self.predictor = FlowerPredictor(model_path, device)

    def predict_directory(
        self, input_dir: str, extensions: List[str] = None
    ) -> List[Dict]:
        """
        Predict flower classes for all images in directory.

        Args:
            input_dir: Directory containing images
            extensions: Allowed file extensions

        Returns:
            List of prediction dictionaries
        """
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")

        print(f"Found {len(image_files)} images to process")

        # Process each image
        results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.predictor.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({"image_path": str(image_path), "error": str(e)})

        return results

    def predict_file_list(self, file_list: List[str]) -> List[Dict]:
        """
        Predict flower classes for list of image files.

        Args:
            file_list: List of image file paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path in tqdm(file_list, desc="Processing images"):
            try:
                result = self.predictor.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({"image_path": str(image_path), "error": str(e)})

        return results


@click.command()
@click.option("--input-dir", help="Directory containing images to predict")
@click.option("--file-list", help="Text file containing list of image paths")
@click.option("--model-path", required=True, help="Path to model checkpoint")
@click.option("--output-file", required=True, help="Output JSON file for results")
@click.option("--device", default="auto", help="Device to use (auto/cpu/cuda)")
def main(
    input_dir: str, file_list: str, model_path: str, output_file: str, device: str
):
    """Predict flower classes for multiple images."""

    if not input_dir and not file_list:
        raise click.UsageError("Either --input-dir or --file-list must be specified")

    if input_dir and file_list:
        raise click.UsageError("Cannot specify both --input-dir and --file-list")

    predictor = BatchFlowerPredictor(model_path, device)

    if input_dir:
        results = predictor.predict_directory(input_dir)
    else:
        with open(file_list, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        results = predictor.predict_file_list(files)

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_file}")
    print(f"Processed {len(results)} images")

    # Print summary
    successful = len([r for r in results if "error" not in r])
    errors = len(results) - successful
    print(f"Successful predictions: {successful}")
    if errors > 0:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
