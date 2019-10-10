"""
----------------------------------------------------------------
Keegan McCluskey
Creates table in Dynamodb to hold post ids and the encodings of their titles
----------------------------------------------------------------
"""

# !/usr/bin/env python
import boto3
import tensorflow as tf
import sys

# Import the Universal Sentence Encoder's TF Hub module
module_url = 'module'
dynamodb = boto3.client('dynamodb')
# Reduce logging output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

table_name = input("Enter Table Name:")
read_units = int(input("Enter Read Capacity Units:"))
write_units = int(input("Enter Write Capacity Units:"))
exit_msg = "Table named: " + table_name + " already exists."

# initialize table
try:
    submission_table = dynamodb.create_table(
        TableName=table_name,
        KeySchema=[
            {
                'AttributeName': 'submission_id',
                'KeyType': 'HASH'
            }
        ],
        AttributeDefinitions=[
            {
                'AttributeName': 'submission_id',
                'AttributeType': 'S'
            }
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': read_units,
            'WriteCapacityUnits': write_units
        }
    )
except dynamodb.exceptions.ResourceInUseException:
    sys.exit(exit_msg)

print("Table named", table_name, "created.")
