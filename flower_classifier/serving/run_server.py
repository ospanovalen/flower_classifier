"""CLI script to run FastAPI inference server."""

import os

import click
import uvicorn


@click.command()
@click.option("--model-path", required=True, help="Path to trained model checkpoint")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, help="Server port")
@click.option("--device", default="auto", help="Device for inference (auto, cpu, cuda)")
@click.option("--workers", default=1, help="Number of worker processes")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
def run_server(
    model_path: str, host: str, port: int, device: str, workers: int, reload: bool
):
    """Run FastAPI inference server for flower classification."""
    print("Starting Flower Classification API server...")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Server: http://{host}:{port}")
    print(f"Workers: {workers}")

    # Set environment variables for the app factory
    os.environ["FLOWER_MODEL_PATH"] = model_path
    os.environ["FLOWER_DEVICE"] = device

    # Run server
    uvicorn.run(
        "flower_classifier.serving.fastapi_server:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
