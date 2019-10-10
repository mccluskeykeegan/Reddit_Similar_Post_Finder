"""
----------------------------------------------------------------
Keegan McCluskey
Tools Used:
    - Reddit API - PRAW
    - PushShift API
    - AWS DynamoDB
    - TensorFlow

Checks post from last 24 hours in given subreddit and looks in dynamodb
database created to hold history of posts from subreddit
for similar posts using Universal Sentence Encoder. Either prints
similar posts to console or prints comments on posts linking
similar posts. Updates databases, maintaining database of posts
with comments to be linked and posts without comments which won't be
linked but, if they are commented on, will be moved to the database
with posts with comments.
----------------------------------------------------------------
"""

# !/usr/bin/env python
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import praw  # not in Lambda
    import os
    import boto3
    from psaw import PushshiftAPI  # not in Lambda
    import tensorflow as tf  # not in Lambda
    import tensorflow_hub as hub  # not in Lambda
    import numpy as np  # not in Lambda
    from decimal import Decimal
    import json
    import operator
    import prawcore.exceptions  # not in Lambda
    import sys


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


def populate_table(dict, name):
    """ add submissions to our DynamoDB table """
    table = get_table(name)

    with table.batch_writer() as batch:
        for i in dict:
            batch.put_item(
                Item={
                    'submission_id': i,
                    'submission_array': dict[i]
                }
            )

    return True


def get_session(module):
    """ Get Universal Encoder session """
    with tf.Graph().as_default():
        sentences = tf.compat.v1.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.compat.v1.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


def get_one_day_subs():
    """ Get submissions from previous day, store ids and encodings in dictionary """
    new_subs = {}
    with_comments = {}
    without_comments = {}
    ids = []
    titles = []

    # Get today's posts from r/learnpython
    for submission_id in api.search_submissions(after="24h",
                                                subreddit=subreddit_name):
        submission = reddit.submission(submission_id)

        # make sure submission isn't stickied, add to list
        if not submission.stickied:
            ids.append(submission.id)
            titles.append(submission.title)

    if len(ids) != 0:
        # 512-element encodings in list form
        encodings = float_to_decimal(embed_fn(titles).tolist())
        for i in range(len(ids)):
            submission = reddit.submission(ids[i])
            new_subs[ids[i]] = encodings[i]
            if submission.num_comments != 0:
                with_comments[ids[i]] = encodings[i]
            else:
                without_comments[ids[i]] = encodings[i]

    return new_subs, with_comments, without_comments  # dict: [id] --> encoding


def table_to_dict(name):
    """ Get entries from table in DB, return list of dictionaries, check for deleted posts and delete from
    table """
    table = get_table(name)
    response = table.scan()
    data = response['Items']

    while response.get('LastEvaluatedKey'):
        response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
        data.extend(response['Items'])

    dict = {}
    for i in range(len(data)):
        try:
            sub = reddit.submission(data[i].get('submission_id'))
            dict[data[i].get('submission_id')] = data[i].get('submission_array')
        except prawcore.exceptions.NotFound:
            # post has been deleted, remove from database
            print(Color.RED, data[i].get('submission_id'), Color.END, 'removed from large_table')
            table.delete_item(
                Key={
                    'submission_id': data[i].get('submission_id')
                }
            )

    return dict  # dict: [id] --> encoding


def calc_inners(new_subs, old_subs):
    """ Calculate inner products between todays posts and large_table, save top 3 matches better than .8 for each new
    sub """
    similars = {}

    for i in new_subs:
        similars[i] = {}

    for i in old_subs:
        for j in new_subs:
            inner = np.inner(old_subs[i], new_subs[j])
            if inner >= .8:
                similars[j][i] = inner

    return similars  # dict: [id] --> [ids]


def create_comment(submission, submissions_to_link):
    """ given id of submission and id of submissions to link, format and post comment """
    post = reddit.submission(submission)
    print(Color.BOLD, Color.YELLOW, post.id, Color.END, post.title)

    reply = ''
    for i in submissions_to_link:
        post_to_link = reddit.submission(i)
        url = post_to_link.url
        title = post_to_link.title
        comments = post_to_link.num_comments
        if comments == 1:
            comment = f'[{title}]({url}) -- 1 comment\n\n'
            print(Color.CYAN, '\t', title, Color.END, '-- 1 comment')
        else:
            comment = f'[{title}]({url}) -- {comments} comments\n\n'
            print(Color.CYAN, '\t', title, Color.END, '--', comments, 'comments')

        reply = reply + comment

    post.reply(reply)
    print()


def make_comments(new_subs, old_subs):
    """ Iterate through similars and write comments to new submission with similar posts"""
    similars = calc_inners(new_subs, old_subs)
    new_similars = {}

    for i in similars:
        new_similars[i] = sorted(similars[i].items(), key=operator.itemgetter(1), reverse=True)[:3]

    for i in new_similars:
        if comment_post:
            create_comment(i, new_similars[i])
        if print_console:
            print('\n', Color.YELLOW, reddit.submission(i).title, Color.END, reddit.submission(i).url)
            for j in new_similars[i]:
                print(reddit.submission(j[0]).title, reddit.submission(j[0]).url)


def iterate_small_table(subs):
    """ move posts with comments to large table """
    with_comments = {}

    for i in subs:
        sub = reddit.submission(i)
        comments = sub.comments
        if len(comments) > 1:
            with_comments[i] = subs[i]
        if len(comments) == 1:
            if comments[0].author != REDDIT_user_id:
                with_comments[i] = subs[i]

    populate_table(with_comments, large_table)


def yes_or_no(question):
    """ prompt for y or n """
    while "the answer is invalid":
        reply = str(input(question + ' (y/n): ')).lower().strip()
        if reply[0] == 'y':
            return True
        if reply[0] == 'n':
            return False


def main():
    new_subs, with_comments, without_comments = get_one_day_subs()
    old_subs = table_to_dict(large_table)
    small_subs = table_to_dict(small_table)

    # go through small table and add posts with comments to large table
    iterate_small_table(small_subs)

    make_comments(new_subs, old_subs)
    # populate_table(with_comments, large_table)
    # populate_table(without_comments, small_table)


# Get resources
dynamodb = boto3.resource('dynamodb')
reddit = get_reddit_credentials()
api = PushshiftAPI(reddit)
embed_fn = get_session("module")

print_console = yes_or_no("print posts to console?")
comment_post = yes_or_no("comment on posts?")
large_table = input("large table name:")
small_table = input("small table name:")
subreddit_name = input("subreddit name:")
main()
