# ir_52

### Information Retrieval Project

This project is a scalable search engine developed as part of the Information Retrieval course at Ben Gurion University, mainly for Software and Information Systems Engineering students. It builds on elements from previous coursework, integrating them into a functional system capable of handling extensive datasets.

### How the Search Engine Works

This search engine is designed to index and search across 6,348,910 Wikipedia pages, with each page uniquely identified by an ID. The engine processes and indexes these pages by separating the content into three main components: ID, Text, and Anchor Text, allowing for efficient retrieval and relevance scoring based on user queries.

Here's the search process in short:

1. Submission: Users submit queries via an HTTP request to the search engine hosted on a GCP VM instance, using the predefined port 8080. The query URL follows the format: http://[VM_INSTANCE_IP]:8080/search?query=[QUERY], where [VM_INSTANCE_IP] is replaced with the IP address of the VM instance, and [QUERY] is replaced with the user's search query.
2. Parsing: The engine parses the user's query, identifying key terms and considering special syntax for exact phrase matches.
3. Index Searching: Utilizing an inverted index, the engine efficiently identifies documents containing the query terms from the extensive Wikipedia dataset, supported by Google Cloud Platform's scalable resources.
4. Scoring and Ranking: Documents are ranked using page views, PageRank scores and relevance scoring algorithms like TF-IDF and BM25, based on the occurrence and distribution of query terms within the Text and Anchor Text.
5. Result Retrieval: Titles of top-ranked documents are returned to the user, with each document's ID linking to a Wikipedia page. These can be accessed directly via https://en.wikipedia.org/?curid=[ID], substituting [ID] with the document's ID.

### Repository Structure

This repository contains all the components necessary for setting up and running the search engine. Below is an overview of the main directories and files included:

- GCP Jupyter Notebooks: This folder houses several Jupyter notebooks that are essential for creating the indexes and necessary dictionaries required by the search engine. These notebooks detail the preprocessing and indexing steps performed on the Wikipedia dataset. Additionally, the inverted_index_gcp.py file within this folder provides utility functions that support index operations in a Google Cloud Platform environment.
- search: Contains the core files of the search engine, including:
  - search_frontend.py: The application that serves as the front end of the search engine, handling HTTP requests and interfacing with the backend.
  - backend.py: Implements the backend logic of the search engine, including query processing, scoring, and ranking of documents.
  - inverted_index_gcp.py: Provides additional utility functions for managing the inverted index.
- buckets_content.txt: A text file listing the contents of the GCP storage bucket used by the search engine.
- graphframes.sh: An initialization script for setting up the Google Cloud Dataproc cluster.
- queries_train.json: A training set of queries, complete with predicted title IDs. This can be used to test and evaluate the search engine's performance and accuracy in retrieving relevant documents.
