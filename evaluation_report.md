# Retrieval evaluation

Precision@5 is the fraction of the five retrieved chunks whose company and page match the label. Recall@5 is 1 when every labeled page appears at least once.

| Question | Mode | Precision@5 | Recall@5 | Latency ms |
|---|---|---:|---:|---:|
| My football boots tore on concrete. Does VelocityX cover that? | semantic | 0.20 | 1.00 | 831 |
| My football boots tore on concrete. Does VelocityX cover that? | hybrid+rerank | 0.20 | 1.00 | 950 |
| The sole came off my VelocityX shoes after three weeks. Is that a warranty case? | semantic | 0.20 | 1.00 | 411 |
| The sole came off my VelocityX shoes after three weeks. Is that a warranty case? | hybrid+rerank | 0.40 | 1.00 | 868 |
| What is the VelocityX standard return window? | semantic | 0.40 | 1.00 | 400 |
| What is the VelocityX standard return window? | hybrid+rerank | 0.20 | 1.00 | 934 |
| What is the VelocityX standard exchange window? | semantic | 0.20 | 1.00 | 411 |
| What is the VelocityX standard exchange window? | hybrid+rerank | 0.40 | 1.00 | 858 |
| Does VelocityX offer prepaid return labels in North America? | semantic | 0.00 | 0.00 | 380 |
| Does VelocityX offer prepaid return labels in North America? | hybrid+rerank | 0.20 | 1.00 | 856 |
| How do VelocityX size exchanges work in India? | semantic | 0.20 | 1.00 | 450 |
| How do VelocityX size exchanges work in India? | hybrid+rerank | 0.20 | 1.00 | 864 |
| How long is the LuxeThreads return window for fine jewelry? | semantic | 0.00 | 0.00 | 396 |
| How long is the LuxeThreads return window for fine jewelry? | hybrid+rerank | 0.20 | 1.00 | 884 |
| How long does a LuxeThreads credit card refund take? | semantic | 0.20 | 1.00 | 422 |
| How long does a LuxeThreads credit card refund take? | hybrid+rerank | 0.40 | 1.00 | 909 |

## Averages

- semantic: precision 0.18, recall 0.75, latency 463 ms
- hybrid+rerank: precision 0.28, recall 1.00, latency 890 ms


## Conclusion

Hybrid search plus reranking recalls every labeled page (1.00) against 0.75 for semantic search. The two semantic misses are VelocityX page 10 (prepaid return labels) and LuxeThreads page 1 (fine jewelry). Average precision rises from 0.18 to 0.28. Hybrid latency is 890 ms against 463 ms for semantic search.