"""Module for downloading flower dataset from Google Drive."""

import shutil
import tarfile
from pathlib import Path

import gdown


def download_data(data_dir: str = "data") -> None:
    """Download and extract flower dataset from Google Drive.

    Args:
        data_dir: Directory to save the data
    """
    data_path = Path(data_dir)
    raw_path = data_path / "raw"

    # Create data directory if it doesn't exist
    data_path.mkdir(exist_ok=True)

    # Skip if data already exists
    if raw_path.exists() and len(list(raw_path.glob("*"))) > 0:
        print(f"Data already exists in {raw_path}")
        return

    # Google Drive file ID for the flower dataset archive
    # From: https://drive.google.com/file/d/1n-DjQGxlEd4iH9skG8xosnfQMSvWm4kf/view
    file_id = "1n-DjQGxlEd4iH9skG8xosnfQMSvWm4kf"

    # Download archive
    archive_path = data_path / "flower_data.tar.gz"
    url = f"https://drive.google.com/uc?id={file_id}"

    print("Downloading flower dataset from Google Drive...")
    print(f"Source: https://drive.google.com/file/d/{file_id}/view")

    try:
        gdown.download(url, str(archive_path), quiet=False)
        print("Download completed successfully")
    except Exception as e:
        print(f"Failed to download from Google Drive: {e}")
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print(f"1. Go to: https://drive.google.com/file/d/{file_id}/view")
        print("2. Click 'Download' button")
        print(f"3. Save as: {archive_path}")
        print("4. Run this script again")
        print("=" * 60)

        if not archive_path.exists():
            return

    # Extract archive
    if archive_path.exists():
        print(f"Extracting data to {data_path}")
        with tarfile.open(archive_path, "r:gz") as tar:
            # Extract to temporary location first
            temp_extract_path = data_path / "temp_extract"
            tar.extractall(temp_extract_path)

            # Move files to correct location
            extracted_raw_path = temp_extract_path / "data" / "raw"
            if extracted_raw_path.exists():
                # Move contents of extracted data/raw to our data/raw
                for item in extracted_raw_path.iterdir():
                    shutil.move(str(item), str(raw_path))
                # Clean up temporary directory
                shutil.rmtree(temp_extract_path)
            else:
                # If structure is different, extract directly
                tar.extractall(data_path)

        # Clean up archive
        archive_path.unlink()
        print(f"Dataset ready in {raw_path}")
    else:
        print("Archive not found. Please download manually.")


if __name__ == "__main__":
    download_data()
