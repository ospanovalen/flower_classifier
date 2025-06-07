"""TensorRT inference module for flower classification."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import click
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)


class TensorRTPredictor:
    """TensorRT inference predictor for flower classification."""

    CLASSES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    def __init__(self, engine_path: str):
        """Initialize TensorRT predictor.

        Args:
            engine_path: Path to TensorRT engine file
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT is not available. Please install TensorRT and "
                "pycuda:\npip install pycuda\nFor TensorRT installation: "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
            )

        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self._load_engine()
        self._allocate_buffers()

    def _load_engine(self):
        """Load TensorRT engine from file."""
        logger.info(f"Loading TensorRT engine from {self.engine_path}")

        # Initialize TensorRT
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        # Load engine
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self.engine_path}"
            )

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        logger.info("TensorRT engine loaded successfully")

    def _allocate_buffers(self):
        """Allocate GPU and CPU buffers for inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to the appropriate list
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "shape": self.engine.get_binding_shape(binding),
                    }
                )
            else:
                self.outputs.append(
                    {
                        "host": host_mem,
                        "device": device_mem,
                        "shape": self.engine.get_binding_shape(binding),
                    }
                )

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: PIL Image

        Returns:
            Preprocessed numpy array
        """
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        tensor = self.transform(image)

        # Convert to numpy and add batch dimension
        array = tensor.numpy()
        return np.expand_dims(array, axis=0)

    def predict(self, image: Image.Image) -> Dict[str, float]:
        """Predict flower class for a single image.

        Args:
            image: PIL Image to classify

        Returns:
            Dictionary with predicted class and probabilities
        """
        # Preprocess image
        input_data = self._preprocess_image(image)

        # Copy input data to GPU
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Copy output data from GPU
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )
        self.stream.synchronize()

        # Process output
        output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        probabilities = torch.softmax(torch.from_numpy(output), dim=1).numpy()[0]

        # Create result dictionary
        result = {
            "predicted_class": self.CLASSES[np.argmax(probabilities)],
            "confidence": float(np.max(probabilities)),
            "all_probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.CLASSES, probabilities)
            },
        }

        return result

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        """Predict flower classes for multiple images.

        Args:
            images: List of PIL Images to classify

        Returns:
            List of prediction dictionaries
        """
        results = []

        for image in images:
            result = self.predict(image)
            results.append(result)

        return results

    def benchmark(self, image: Image.Image, iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance.

        Args:
            image: PIL Image for benchmarking
            iterations: Number of iterations to run

        Returns:
            Performance statistics
        """
        # Warmup
        for _ in range(10):
            self.predict(image)

        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            self.predict(image)
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / iterations
        throughput = iterations / total_time

        return {
            "total_time": total_time,
            "avg_time_per_image": avg_time,
            "images_per_second": throughput,
            "iterations": iterations,
        }

    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, "stream") and self.stream:
            self.stream.synchronize()


@click.command()
@click.option("--engine-path", required=True, help="Path to TensorRT engine file")
@click.option("--image-path", required=True, help="Path to input image")
@click.option("--output-file", help="Path to save prediction results (JSON)")
@click.option("--benchmark", is_flag=True, help="Run benchmark test")
@click.option("--iterations", default=100, help="Number of benchmark iterations")
def main(
    engine_path: str,
    image_path: str,
    output_file: Optional[str],
    benchmark: bool,
    iterations: int,
):
    """TensorRT inference CLI for flower classification."""
    try:
        # Load predictor
        predictor = TensorRTPredictor(engine_path)

        # Load image
        image = Image.open(image_path)

        if benchmark:
            # Run benchmark
            logger.info(f"Running benchmark with {iterations} iterations...")
            stats = predictor.benchmark(image, iterations)

            print("\n🚀 TensorRT Benchmark Results:")
            print(f"⏱️  Total time: {stats['total_time']:.3f}s")
            print(
                f"📊 Avg time per image: " f"{stats['avg_time_per_image']*1000:.2f}ms"
            )
            print(f"🔥 Throughput: " f"{stats['images_per_second']:.1f} images/sec")

        # Make prediction
        result = predictor.predict(image)

        print(f"\n🌸 Prediction for {image_path}:")
        print(f"🎯 Class: {result['predicted_class']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print("\n📈 All probabilities:")
        for class_name, prob in result["all_probabilities"].items():
            print(f"   {class_name}: {prob:.3f}")

        # Save results if requested
        if output_file:
            import json

            result["image_path"] = image_path
            result["engine_path"] = engine_path

            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Results saved to {output_file}")

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
