# Proposal

Text Embedding Pipeline:

1. Text chunking
2. Embed each chunk as vectors: dense and spare
3. Store embedding as the key and put into graph

Retrieval Pipeline:

1. Embed prompt as vector
2. Sparse and Dense retrievers (graph-based search)
3. Reranking of chunks
4. Context Relevance filter + Context Usefulness filter
5. Context + Prompt => generator => answer 