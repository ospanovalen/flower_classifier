from pathlib import Path
from typing import Tuple

import click
import torch
import torch.onnx

from flower_classifier.models.flower_model import FlowerClassifier


class ONNXConverter:
    """Convert PyTorch Lightning model to ONNX format."""

    def __init__(self, checkpoint_path: str):
        """
        Initialize converter.

        Args:
            checkpoint_path: Path to Lightning checkpoint
        """
        self.checkpoint_path = checkpoint_path
        self.model = self._load_model()

    def _load_model(self) -> FlowerClassifier:
        """Load model from checkpoint."""
        model = FlowerClassifier.load_from_checkpoint(self.checkpoint_path)
        model.eval()
        # Move model to CPU for ONNX export to avoid device mismatch
        model = model.cpu()
        return model

    def convert(
        self,
        output_path: str,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 11,
        dynamic_axes: bool = True,
    ) -> None:
        """
        Convert model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch, channels, height, width)
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic batch size
        """
        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Set up dynamic axes for variable batch size
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting model from {self.checkpoint_path}")
        print(f"Input shape: {input_shape}")
        print(f"Output path: {output_path}")
        print(f"Opset version: {opset_version}")
        print(f"Dynamic axes: {dynamic_axes}")

        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes_dict,
            verbose=True,
        )

        print(f"Model successfully converted to ONNX: {output_path}")

        # Verify the exported model
        self._verify_onnx_model(output_path, dummy_input)

    def _verify_onnx_model(self, onnx_path: Path, dummy_input: torch.Tensor) -> None:
        """
        Verify that the ONNX model produces the same output as PyTorch model.

        Args:
            onnx_path: Path to ONNX model
            dummy_input: Input tensor for verification
        """
        try:
            import onnx
            import onnxruntime as ort
        except ImportError:
            print("ONNX verification skipped: onnx/onnxruntime not installed")
            print("   Install with: pip install onnx onnxruntime")
            return

        # Load ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(str(onnx_path))

        # Get PyTorch model output
        with torch.no_grad():
            # Ensure model and input are on the same device (CPU)
            model_cpu = self.model.cpu()
            input_cpu = dummy_input.cpu()
            pytorch_output = model_cpu(input_cpu).numpy()

        # Get ONNX model output
        onnx_output = ort_session.run(None, {"input": dummy_input.numpy()})[0]

        # Compare outputs
        diff = abs(pytorch_output - onnx_output).max()
        if diff < 1e-5:
            print(f"ONNX model verification passed (max diff: {diff:.2e})")
        else:
            print(f"ONNX model verification failed (max diff: {diff:.2e})")


@click.command()
@click.option("--checkpoint-path", required=True, help="Path to Lightning checkpoint")
@click.option("--output-path", required=True, help="Path to save ONNX model")
@click.option(
    "--input-shape", default="1,3,224,224", help="Input shape (comma-separated)"
)
@click.option("--opset-version", default=11, help="ONNX opset version")
@click.option("--no-dynamic-axes", is_flag=True, help="Disable dynamic batch size")
def main(
    checkpoint_path: str,
    output_path: str,
    input_shape: str,
    opset_version: int,
    no_dynamic_axes: bool,
):
    """Convert PyTorch Lightning checkpoint to ONNX format."""

    # Parse input shape
    shape = tuple(map(int, input_shape.split(",")))
    if len(shape) != 4:
        raise click.BadParameter(
            "Input shape must have 4 dimensions (batch,channels,height,width)"
        )

    # Convert model
    converter = ONNXConverter(checkpoint_path)
    converter.convert(
        output_path=output_path,
        input_shape=shape,
        opset_version=opset_version,
        dynamic_axes=not no_dynamic_axes,
    )


if __name__ == "__main__":
    main()
