\# SQLite Performance and Search Notes



\## WAL Mode and Pragmas

Enabling Write-Ahead Logging (`PRAGMA journal\_mode=WAL;`) allows concurrent reads during write transactions. Pairing it with `PRAGMA synchronous=NORMAL;` significantly speeds up batch inserts while remaining safe against application crashes.



\## Hybrid Retrieval Architecture

Combining `FTS5` full-text search (BM25 ranking) with `sqlite-vec` vector similarity (`int8` quantized embeddings) solves two complementary retrieval problems: exact keyword matching for identifiers and semantic matching for conceptual queries. Reciprocal Rank Fusion (RRF) merges both ranked lists without requiring score normalization.



