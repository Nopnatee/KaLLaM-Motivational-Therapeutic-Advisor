import os
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_aws_credentials():
    try:
        # Test AWS credentials
        print("Testing AWS credentials...")
        
        # Print current environment variables (without revealing secrets)
        print(f"AWS_REGION: {os.getenv('AWS_REGION', 'Not set')}")
        print(f"AWS_ACCESS_KEY_ID: {'Set' if os.getenv('AWS_ACCESS_KEY_ID') else 'Not set'}")
        print(f"AWS_SECRET_ACCESS_KEY: {'Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else 'Not set'}")
        print(f"AWS_SESSION_TOKEN: {'Set' if os.getenv('AWS_SESSION_TOKEN') else 'Not set'}")
        
        # Try to create a session and get caller identity
        session = boto3.Session(
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_session_token=os.getenv('AWS_SESSION_TOKEN'),
            region_name=os.getenv('AWS_REGION', 'ap-southeast-2')
        )
        
        sts = session.client('sts')
        identity = sts.get_caller_identity()
        
        print("✅ AWS credentials are valid!")
        print(f"Account: {identity.get('Account')}")
        print(f"UserId: {identity.get('UserId')}")
        print(f"Arn: {identity.get('Arn')}")
        
        # Test Bedrock access specifically
        bedrock = session.client('bedrock', region_name=os.getenv('AWS_REGION', 'ap-southeast-2'))
        models = bedrock.list_foundation_models()
        print("✅ Bedrock access confirmed!")
        print(f"Available models: {len(models.get('modelSummaries', []))}")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS credentials test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_aws_credentials()