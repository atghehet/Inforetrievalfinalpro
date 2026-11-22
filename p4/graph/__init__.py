"""Graph algorithms on the citation graph built by P1 (edges.parquet).

    edges.py       read and clean the edge table (shared by both algorithms)
    pagerank.py    PageRank: which papers are important in the citation network
    cocitation.py  co-citation recommender: which papers are cited together with a given paper
    cli.py         command-line interface for both (python -m p4.graph.cli --help)

The two algorithms are independent: neither imports the other.
"""
