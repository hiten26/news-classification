import pickle
import argparse
import boto3
import sklearn

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--id", help="Key ID")
parser.add_argument("-a", "--access", help=" Access Value")

args = parser.parse_args()

session = boto3.Session(
    aws_access_key_id=args.id,
    aws_secret_access_key=args.access
)

s3 = session.resource('s3')

model = pickle.loads(
    s3.Bucket("classification-news").Object("models/logistic_regression_model.pkl").get()['Body'].read())
transform = pickle.loads(s3.Bucket("classification-news").Object("models/transform.pkl").get()['Body'].read())

pickle.dump(model, open('models/logistic_regression_model.pkl', 'wb'))
pickle.dump(transform, open('models/transform.pkl', 'wb'))
