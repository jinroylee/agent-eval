## 1. 공통

- LLM-as-a-judge
    - GT (정답 필요)/non-GT (정답 불필요)
    - 평가 기준(Criteria): Completeness(완전성), Clarity(명확성), Usefulness(유용성), Relevance(관련성) + Friendliness(친근함)..?
    - PoLL에 대한 추가 설명
    - 필요한 상태 필드:
        - 사용자 메시지(Input)
        - 최종 응답(Predicted)
- BERTScore
    - GT
    - 필요한 상태 필드
        - 최종 응답(Predicted)
- P95 Latency
    - non-GT
    - 필요 필드:
        - 없음
- 소비 토큰(Token spent)
    - non-GT
    - 입력 상태(Input State):
    - 필요 필드:
        - 소비 토큰(Token spent)

## 2. RAG

- Recall@K
    - GT
    - 필요 필드:
        - k (평가를 위해 k를 고정한다; 따라서 k를 에이전트 입력으로 주입할 수 있어야 한다.)
        - 검색된 청크(Retrieved Chunks)
- Precision@K
    - GT
    - 필요 필드:
        - k
        - 검색된 청크(Retrieved Chunks)
- NDCG@k
    - GT
    - 필요 필드:
        - k
        - 검색된 청크(Retrieved Chunks)
- faithfulness
    - non-GT
    - 필요 필드:
        - 검색된 청크(Retrieved Chunks)
        - 최종 응답
- consistency
    - non-GT
    - 필요 필드:
        - 검색된 청크(Retrieved Chunks)
        - 최종 응답

## 3. T2S

- Soft F1
    - GT
    - 필요 필드:
        - 실행 결과(Execution Result)
- Component Match
    - GT
    - 필요 필드:
        - 생성된 SQL(Generated SQL)
- T2S Faithfulness
    - non-GT
    - 필요 필드:
        - 실행 결과(Execution Result)
        - 최종 응답
- AST Validity
    - non-GT
    - 필요 필드:
        - 생성된 SQL(Generated SQL)
- consistency
    - non-GT
    - 필요 필드:
        - 실행 결과(Execution Result)
        - 최종 응답
