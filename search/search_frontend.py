from flask import Flask, request, jsonify
from google.cloud import storage
import io
from io import BytesIO
import pickle
import gzip
import pandas as pd
from inverted_index_gcp import *
from backend import *
import re

# WILL NEED TO IMPLEMENT backend.py
bucket_name = "ir_52"

# Path indexes
path_index = "index"
path_title_pkl = "title_stemmed"
path_text_pkl = "text_stemmed"
path_anchor_pkl = "anchor_stemmed"

# Path dictionaries
path_id_title_1_pkl = "id_title/id_title_1.pkl"
path_id_title_2_pkl = "id_title/id_title_2.pkl"
path_doc_lengths_pkl = "doc_lengths/doc_lengths.pkl"
path_page_views_pkl = "page_views/pageviews-202108-user.pkl"
path_page_ranks_gz = "page_ranks/pr_part-00000-ea0a3e01-99d0-489b-be3e-ccae70afa81a-c000.csv.gz"

# Read Indices
inverted_title = InvertedIndex.read_index(path_title_pkl, path_index, bucket_name)
inverted_text = InvertedIndex.read_index(path_text_pkl, path_index, bucket_name)
inverted_anchor = InvertedIndex.read_index(path_anchor_pkl, path_index, bucket_name)

def get_bucket(bucket_name):
    """Retrieve a Google Cloud Storage bucket."""
    client = storage.Client()
    return client.bucket(bucket_name)

def download_blob_as_bytes(bucket, path):
    """Download a blob from the specified bucket as bytes."""
    blob = bucket.blob(path)
    return blob.download_as_bytes()

bucket = get_bucket(bucket_name)

id_title_1_bytes = download_blob_as_bytes(bucket, path_id_title_1_pkl)
id_title_1 = pickle.loads(id_title_1_bytes)

# Download and load id_title_2.pkl
id_title_2_bytes = download_blob_as_bytes(bucket, path_id_title_2_pkl)
id_title_2 = pickle.loads(id_title_2_bytes)

doc_lengths_bytes = download_blob_as_bytes(bucket, path_doc_lengths_pkl)
doc_lengths = pickle.loads(doc_lengths_bytes)
doc_N = len(doc_lengths)
avg_doc_length = sum(doc_lengths.values()) / doc_N

# Download and load page_views.pkl
page_views_bytes = download_blob_as_bytes(bucket, path_page_views_pkl)
page_views = pickle.loads(page_views_bytes)
views_max = max(page_views.values())
page_views_norm = {id: view / views_max for id, view in page_views.items()}

# Download and process page_ranks.gz
page_ranks_gz_bytes = download_blob_as_bytes(bucket, path_page_ranks_gz)

# Decompress and read the CSV data
with gzip.GzipFile(fileobj=BytesIO(page_ranks_gz_bytes)) as f:
    page_ranks = pd.read_csv(f, header=None, index_col=0).squeeze("columns").to_dict()
ranks_max = max(page_ranks.values())
page_ranks_norm = {id: rank / ranks_max for id, rank in page_ranks.items()}

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    final_score = Counter()
    top_results = 500

    tokens = tokenize(query)

    # Title (Binary score)
    title_scores = word_count_score(tokens, inverted_title)[:top_results]
    n_tokens = len(tokens)
    title_norm = [(pair[0], pair[1]/n_tokens) for pair in title_scores]

    # Text (BM25 score)
    text_scores = BM25_score(tokens, inverted_text, doc_N, doc_lengths, avg_doc_length)[:top_results]
    text_max = text_scores[0][1]
    text_norm = [(pair[0], pair[1]/text_max) for pair in text_scores]

    # Anchor (TF score)
    anchor_scores = tf_score(tokens, inverted_anchor)[:top_results]
    anchor_max = anchor_scores[0][1]
    anchor_norm = [(pair[0], pair[1]/anchor_max) for pair in anchor_scores]



    w_title = 1.5
    w_text = 1.5
    w_anchor = 2
    pr = 1
    pv = 1
    if "?" in query:
      w_text = w_text*1.2
      w_anchor = w_anchor *1.2
    # Combine normalized scores
    for norm, weight in ((title_norm, w_title), (text_norm, w_text), (anchor_norm, w_anchor)):
        for id, score in norm:
            if id not in final_score:
                final_score[id] = 0
            # print(score, weight)
            final_score[id] += score * weight

    # PageRank and PageViews scores

    for id in final_score:
        if id in page_ranks:
            final_score[id] += page_ranks_norm[id] * pr
        if id in page_views:
            final_score[id] += page_views_norm[id] * pv
                    
    
    # Sort and take the top 100
    final_list = final_score.most_common()
    # std and mean
    scores = [pair[1] for pair in final_list]
    mean = sum(scores)/len(scores)
    std = (sum([(score - mean)**2 for score in scores])/len(scores))**0.5
    # filter out results with score below (mean + 2*std)
    # final_list_filtered = final_list.filter(lambda x: x[1] > (mean + 2*std))
    # final_list_filtered = [pair for pair in final_list if pair[1] > (mean + 1.2*std)]
    final_list_filtered = [id for i, (id, score) in enumerate(final_list) if score > (mean + 1.15*std) or i < 5][:100]
    # get result titles
    # res = [(id, id_title[id]) for id in final_list_filtered]
    for id in final_list_filtered:
        if id in id_title_1:
          res.append((str(id), id_title_1[id]))
        if id in id_title_2:
          res.append((str(id), id_title_2[id]))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
