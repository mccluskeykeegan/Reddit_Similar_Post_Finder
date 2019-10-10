"""
----------------------------------------------------------------
Keegan McCluskey
Fills table in Dynamodb with post ids and the encodings of their titles from subreddit
----------------------------------------------------------------
"""

# !/usr/bin/env python
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import praw
    import os
    import boto3
    from psaw import PushshiftAPI
    import tensorflow as tf
    import tensorflow_hub as hub
    from decimal import Decimal
    import json


class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'


# Get environment variables for credentials
REDDIT_client_id = os.getenv('REDDIT_client_id')
REDDIT_client_secret = os.getenv('REDDIT_client_secret')
REDDIT_password = os.getenv('REDDIT_password')
REDDIT_user_id = os.getenv('REDDIT_user_id')
REDDIT_user_name = os.getenv('REDDIT_user_name')


def get_reddit_credentials():
    """ Get Reddit Credentials """
    reddit = praw.Reddit(client_id=REDDIT_client_id,
                         client_secret=REDDIT_client_secret,
                         password=REDDIT_password,
                         user_agent='testscript',
                         username=REDDIT_user_name)
    return reddit


def get_table(name):
    """ Return table from DynamoDB """
    submission_table = dynamodb.Table(name)
    return submission_table


def float_to_decimal(encodings):
    new_encodings = json.loads(json.dumps(encodings), parse_float=Decimal)
    return new_encodings


def populate_table():
    """ add submissions to our DynamoDB table """
    table = get_table(table_name)
    dict = get_reddit_questions()

    with table.batch_writer() as batch:
        for i in dict:
            batch.put_item(
                Item={
                    'submission_id': i,
                    'submission_array': dict[i]
                }
            )


def get_session(module):
    """ Get Universal Encoder session """
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def get_reddit_questions():
    """ Check r/redditdev for this month's posts, add to list """
    # to hold submissions for batch writing
    ids = []
    titles = []
    dict = {}
    cnt = 0

    # Get today's posts from r/learnpython
    for submission_id in api.search_submissions(after=time_after,
                                                before=time_before,
                                                subreddit=subreddit_name):
        submission = reddit.submission(submission_id)

        # make sure submission isn't stickied, add to list
        if not submission.stickied:
            if submission.num_comments != 0:
                ids.append(submission.id)
                titles.append(submission.title)
        cnt += 1

    if cnt != 0:
        encodings = float_to_decimal(embed_fn(titles).tolist())
        for i in range(len(ids)):
            dict[ids[i]] = encodings[i]

    print(cnt, "posts added to table.")
    return dict  # dict: [id] --> encoding


# Get resources
dynamodb = boto3.resource('dynamodb')
reddit = get_reddit_credentials()
api = PushshiftAPI(reddit)
embed_fn = get_session("module")

table_name = input("Enter Table Name:")
time_after = input("time after (+ m, h, d):")
time_before = input("time before (+ m, h, d):")
subreddit_name = input("subreddit name:")

populate_table()
