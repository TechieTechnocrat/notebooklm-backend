import boto3
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET_NAME

s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

def upload_to_s3(file_path, filename):
    s3.upload_file(file_path, S3_BUCKET_NAME, filename)
    url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{filename}"
    return url
