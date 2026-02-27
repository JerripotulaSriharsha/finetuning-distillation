import boto3
import os
from const import AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY
from botocore.client import Config
from botocore.exceptions import NoCredentialsError, ClientError

def sync_data_from_s3(s3_path, local_path="/tmp/data"):
    """
    Sync data from S3/MinIO to local storage.
    
    Args:
        s3_path: S3 path in format s3://bucket-name/path/to/file
        local_path: Local directory to sync data to
    
    Returns:
        Local path to the synced data
    """
    # Extract bucket and key from S3 path
    s3_path = s3_path.replace("s3://", "")
    bucket_name = s3_path.split("/")[0]
    object_key = "/".join(s3_path.split("/")[1:])
    
    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)
    local_file_path = os.path.join(local_path, os.path.basename(object_key))
    
    try:
        # Setup S3 client with credentials
        s3 = boto3.client('s3', 
                         endpoint_url=AWS_ENDPOINT_URL,
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                         config=Config(signature_version='s3v4'))
        
        # Check if object exists before downloading
        try:
            s3.head_object(Bucket=bucket_name, Key=object_key)
        except:
            raise Exception(f"Object {object_key} does not exist in bucket {bucket_name}")
        
        # Download the file
        print(f"Downloading {s3_path} to {local_file_path}")
        s3.download_file(bucket_name, object_key, local_file_path)
        print(f"Successfully downloaded data to {local_file_path}")
        
        return local_file_path
    except NoCredentialsError:
        print("Credentials not available. Please check your AWS credentials.")
        raise
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        raise