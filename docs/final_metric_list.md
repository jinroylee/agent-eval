## 1. 공통

- LLM-as-a-judge
    - GT/non-GT
    - Criteria: Completeness, Clarity, Usefulness, Relevance + Friendliness..?
    - More description on PoLL
    - Required state fields:
        - User message (Input)
        - Final Response (Predicted)
- BERTScore
    - GT
    - Required state fields
        - Final Response (Predicted)
- P95 Latency
    - non-GT
    - Required fields:
        - None
- Token spent
    - non-GT
    - Input State:
    - Required fields:
        - Token spent

## 2. RAG

- Recall@K
    - GT
    - Required fields:
        - k (we fix the number k for evaluation. Therefore, we should be able to inject k as an agent input)
        - Retrieved Chunks
- Precision@K
    - GT
    - Required fields:
        - k
        - Retrieved Chunks
- NDCG@k
    - GT
    - Required fields:
        - k
        - Retrieved Chunks
- faithfulness
    - non-GT
    - Required fields:
        - Retrieved Chunks
        - Final Response
- consistency
    - non-GT
    - Required fields:
        - Repeated Final Responses (same query, N runs)

## 3. T2S

- Soft F1
    - GT
    - Required fields:
        - Execution Result
- Component Match
    - GT
    - Required fields:
        - Generated SQL
- T2S Faithfulness
    - non-GT
    - Required fields:
        - Execution Result
        - Final Response
- AST Validity
    - non-GT
    - Required fields:
        - Generated SQL
- consistency
    - non-GT
    - Required fields:
        - Repeated Generated SQL (same query, N runs)