"""
Upload trained models to S3 bucket.
Uploads Fast Lane model (Logistic Regression) and scaler to S3.
"""

import boto3
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def upload_models(bucket_name: str, model_dir: str):
    """
    Upload models to S3.

    Args:
        bucket_name: S3 bucket name
        model_dir: Local directory containing models
    """
    s3 = boto3.client('s3')

    # Files to upload
    files_to_upload = {
        'fast_lane/lr_model.pkl': f'{model_dir}/lr_model.pkl',
        'fast_lane/scaler.pkl': f'{model_dir}/scaler.pkl'
    }

    print(f"\n{'='*80}")
    print("Uploading Models to S3")
    print(f"{'='*80}\n")
    print(f"Bucket: {bucket_name}")
    print(f"Model directory: {model_dir}\n")

    for s3_key, local_path in files_to_upload.items():
        if not os.path.exists(local_path):
            print(f"❌ File not found: {local_path}")
            continue

        file_size = os.path.getsize(local_path) / 1024 / 1024  # MB

        print(f"Uploading {local_path} ({file_size:.2f} MB) to s3://{bucket_name}/{s3_key}...")

        try:
            s3.upload_file(
                local_path,
                bucket_name,
                s3_key,
                ExtraArgs={
                    'ServerSideEncryption': 'AES256',
                    'Metadata': {
                        'project': 'vpbank-streamguard',
                        'model_type': 'fast_lane',
                        'uploaded_from': 'local'
                    }
                }
            )
            print(f"✓ Uploaded successfully\n")

        except Exception as e:
            print(f"❌ Upload failed: {str(e)}\n")
            raise

    # Verify uploads
    print(f"{'='*80}")
    print("Verifying Uploads")
    print(f"{'='*80}\n")

    for s3_key in files_to_upload.keys():
        try:
            response = s3.head_object(Bucket=bucket_name, Key=s3_key)
            size_mb = response['ContentLength'] / 1024 / 1024
            print(f"✓ s3://{bucket_name}/{s3_key} ({size_mb:.2f} MB)")
        except:
            print(f"❌ s3://{bucket_name}/{s3_key} - NOT FOUND")

    print(f"\n{'='*80}")
    print("Upload Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload models to S3")
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--model-dir', required=True, help='Local model directory')

    args = parser.parse_args()

    upload_models(args.bucket, args.model_dir)
