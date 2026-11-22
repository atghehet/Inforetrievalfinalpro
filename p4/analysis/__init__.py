"""P4 analyses of retrieval results: popularity bias, cold start, publication year, citation function, errors.

    data.py         read P1's tables (queries, papers, context_functions, index_contexts) and join them per query
    groups.py       group definitions (citation buckets, years, functions), per-group summaries, paired comparisons, trends
    popularity.py   retrieval-side popularity bias (how popular are the papers a system returns?)
    errors.py       query-level failures, missed papers, differences between systems
    analyses.py     the five analyses, each returning tables plus a Markdown report
    cli.py          python -m p4.analysis.cli {popularity,coldstart,year,function,errors} ...

Every analysis needs an explicit data label (real / synthetic / demo). The label is declared by the user and is not
verified. Results describe ASSOCIATIONS in the data; nothing here establishes cause and effect.
"""
