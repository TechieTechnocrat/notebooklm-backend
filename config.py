from dotenv import load_dotenv
load_dotenv()

import os


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
