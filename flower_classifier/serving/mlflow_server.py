"""MLflow serving module for flower classification."""

import json
import logging
import subprocess
import time
from typing import Dict, List, Optional

import click
import requests

logger = logging.getLogger(__name__)


class FlowerMLflowServer:
    """MLflow server manager for flower classification."""

    def __init__(self, model_uri: str, host: str = "127.0.0.1", port: int = 5001):
        """Initialize MLflow server.

        Args:
            model_uri: MLflow model URI (e.g., models:/flower_classifier/1)
            host: Server host
            port: Server port
        """
        self.model_uri = model_uri
        self.host = host
        self.port = port
        self.server_process = None
        self.server_url = f"http://{host}:{port}"

    def start_server(self, workers: int = 1, timeout: int = 60):
        """Start MLflow model server.

        Args:
            workers: Number of worker processes
            timeout: Timeout in seconds to wait for server startup
        """
        logger.info(f"Starting MLflow server for model: {self.model_uri}")

        # Build command
        cmd = [
            "mlflow",
            "models",
            "serve",
            "-m",
            self.model_uri,
            "-h",
            self.host,
            "-p",
            str(self.port),
            "--workers",
            str(workers),
            "--no-conda",
        ]

        # Start server process
        try:
            self.server_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Wait for server to start
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{self.server_url}/health")
                    if response.status_code == 200:
                        logger.info(
                            f"MLflow server started successfully at {self.server_url}"
                        )
                        return True
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
                    continue

            # Timeout reached
            logger.error("Timeout waiting for MLflow server to start")
            self.stop_server()
            return False

        except Exception as e:
            logger.error(f"Failed to start MLflow server: {e}")
            return False

    def stop_server(self):
        """Stop MLflow server."""
        if self.server_process:
            logger.info("Stopping MLflow server...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
            self.server_process = None
            logger.info("MLflow server stopped")

    def predict_single(self, image_path: str) -> Dict:
        """Make prediction for single image.

        Args:
            image_path: Path to image file

        Returns:
            Prediction result
        """
        if not self.is_server_running():
            raise RuntimeError("MLflow server is not running")

        # Note: For MLflow serving, we need to adapt input format
        # This is a simplified example - real implementation would need
        # proper image preprocessing to match model expectations
        data = {"instances": [{"image_path": str(image_path)}]}

        # Make prediction request
        try:
            response = requests.post(
                f"{self.server_url}/invocations",
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Make batch predictions.

        Args:
            image_paths: List of image file paths

        Returns:
            List of prediction results
        """
        if not self.is_server_running():
            raise RuntimeError("MLflow server is not running")

        # Prepare batch data
        instances = [{"image_path": str(path)} for path in image_paths]
        data = {"instances": instances}

        # Make prediction request
        try:
            response = requests.post(
                f"{self.server_url}/invocations",
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise

    def is_server_running(self) -> bool:
        """Check if server is running.

        Returns:
            True if server is running
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_server_info(self) -> Dict:
        """Get server information.

        Returns:
            Server info dict
        """
        if not self.is_server_running():
            return {"status": "stopped"}

        try:
            response = requests.get(f"{self.server_url}/version", timeout=5)
            version_info = response.json() if response.status_code == 200 else {}

            return {
                "status": "running",
                "url": self.server_url,
                "model_uri": self.model_uri,
                "version_info": version_info,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


@click.group()
def cli():
    """MLflow serving CLI for flower classification."""
    pass


@cli.command()
@click.option("--model-uri", required=True, help="MLflow model URI")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=5001, help="Server port")
@click.option("--workers", default=1, help="Number of workers")
@click.option("--timeout", default=60, help="Startup timeout in seconds")
def start(model_uri: str, host: str, port: int, workers: int, timeout: int):
    """Start MLflow model server."""
    server = FlowerMLflowServer(model_uri, host, port)

    try:
        success = server.start_server(workers, timeout)
        if success:
            print(f"🚀 Server started at {server.server_url}")
            print("Press Ctrl+C to stop...")

            # Keep server running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n  Stopping server...")
        else:
            print(" Failed to start server")
            exit(1)
    finally:
        server.stop_server()


@cli.command()
@click.option("--server-url", required=True, help="MLflow server URL")
@click.option("--image-path", required=True, help="Path to image")
@click.option("--output-file", help="Output JSON file")
def predict(server_url: str, image_path: str, output_file: Optional[str]):
    """Make prediction using MLflow server."""
    try:
        # Simple prediction request
        data = {"instances": [{"image_path": image_path}]}

        response = requests.post(
            f"{server_url}/invocations",
            headers={"Content-Type": "application/json"},
            json=data,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()

        print(f" Prediction for {image_path}:")
        print(json.dumps(result, indent=2))

        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to {output_file}")

    except Exception as e:
        print(f" Error: {e}")
        exit(1)


@cli.command()
@click.option("--server-url", required=True, help="MLflow server URL")
def status(server_url: str):
    """Check server status."""
    try:
        # Health check
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f" Server is running at {server_url}")

            # Try to get version info
            try:
                version_response = requests.get(f"{server_url}/version", timeout=5)
                if version_response.status_code == 200:
                    version_info = version_response.json()
                    print(" Server info:")
                    print(json.dumps(version_info, indent=2))
            except Exception:
                pass
        else:
            print(f" Server returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f" Cannot connect to server at {server_url}")
    except Exception as e:
        print(f" Error checking server: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
