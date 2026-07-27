# 메트릭 API 서버

입력/출력을 채점하는 모든 메트릭을 HTTP 엔드포인트로 제공하므로, 다른 서비스가 프레임워크를 내장하지
않고도 에이전트 응답을 채점할 수 있다. 서버는 얇은 래퍼(thin wrapper)다: 메트릭의 의미(semantics),
통계, judge 동작은 라이브러리의 것과 정확히 같다(`registry` → `metric.score` → `evaluate()` 집계).

```
POST /{family}/{metric}   ←  1..N EvalContexts (JSON)
                          →  per-item results + ONE aggregated value with a 95% CI
```

`p95_latency`와 `token_usage`에는 **엔드포인트가 없다** — 이 둘은 하니스가 만들어 낸 메타데이터를
읽을 뿐이므로(채점할 입력/출력이 없다) CLI/라이브러리 쪽에 남는다.

## 설치 및 실행

```bash
uv sync --extra server --extra t2s      # or: pip install 'agent-eval[server,t2s]'
uv run agent-eval-serve                 # 127.0.0.1:8000; --host/--port/--workers
```

로컬 실행 시 `agent-eval-serve`는 현재 디렉터리에 `./.env` 파일(`KEY=VALUE`, 이미 설정된
환경변수가 우선)이 있으면 함께 읽는다 — 다른 경로는 `--env-file`로 지정하고, `--env-file ''`로
끌 수 있다.

Docker:

```bash
docker build -t agent-eval-api .
docker run -p 8000:8000 -e AGENT_EVAL_JUDGE_BASE_URL=... -e AGENT_EVAL_JUDGE_MODEL=... agent-eval-api
# BERTScore baked in (pulls torch, multi-GB):
docker build --build-arg EXTRAS="server,t2s,bertscore" -t agent-eval-api .
```

대화형 API 문서는 `/docs`에서, 기계가 읽을 수 있는 메트릭 맵은 `GET /catalog`에서 확인할 수 있다.
라이브니스(liveness), 현재 활성인 judge, 그리고 선택 기능(`t2s`/`bertscore`)의 사용 가능 여부는
`GET /health`가 알려준다.

## 엔드포인트

| 경로 | 메트릭 | 각 컨텍스트에 필요한 값 | `params` |
|---|---|---|---|
| `POST /common/llm_judge` | `llm_judge` | `input`, `output` (선택: `expected`) | `criteria`, `scale`, `system_prompt` |
| `POST /common/bertscore` | `bertscore` | `output`, `expected` | `lang`, `model_type`, `rescale_with_baseline` |
| `POST /rag/recall_at_k` | `recall_at_k` | `metadata.retrieved_ids`, `.relevant_ids` | `k` |
| `POST /rag/precision_at_k` | `precision_at_k` | `metadata.retrieved_ids`, `.relevant_ids` | `k` |
| `POST /rag/ndcg_at_k` | `ndcg_at_k` | 동일 (+ 선택: `.relevance`) | `k` |
| `POST /rag/faithfulness` | `faithfulness` | `output`, `retrieved_context` | `system_prompt` |
| `POST /rag/consistency` | `consistency` | `metadata.repeated_outputs` (선택: `input`) | `system_prompt` |
| `POST /t2s/soft_f1` | `soft_f1` | `metadata.execution_result`, `.gold_execution_result` | `result_policy` |
| `POST /t2s/component_match` | `component_match` | `metadata.sql`, `.gold_sql` | `dialect` |
| `POST /t2s/ast_valid` | `ast_valid` | `metadata.sql` | `dialect` |
| `POST /t2s/faithfulness` | `t2s_faithfulness` | `output`, `metadata.execution_result` (선택: `input`) | `sample_rows`, `max_distinct`, `max_columns`, `system_prompt` |
| `POST /t2s/consistency` | `t2s_consistency` | `metadata.repeated_sql` (선택: `input`) | `system_prompt` |

`result_policy`는 `ResultSetPolicy`의 필드를 그대로 받는다: `row_order`(`ignore|strict`),
`duplicates`(`keep|dedup`), `nulls`(`distinct|coalesce`), `column_order`(`ignore|strict`),
`float_tolerance`. `criteria`(llm_judge)는 기준별 판정(기본 동작)을 위한 리스트 — 각 항목은 이름
문자열 또는 `{name, description}` — 이거나, 종합 점수 1회 호출을 위한 일반 문자열 루브릭이다.
`system_prompt`(judge 엔드포인트)는 이 요청에 한해 judge 페르소나 서문을 교체한다
([judges.md](judges.md) 참고). 어떤 파라미터든 JSON `null`을 명시적으로 주면 "기본값을 사용한다"는
뜻이며, 키를 생략한 것과 같다.

## 요청 / 응답

모든 엔드포인트가 하나의 요청 형태를 공유한다 — 컨텍스트를 하나만 보내도 되고, N개(질문당 하나)를 보낸
뒤 최종 집계값만 읽어도 된다. 일관성(consistency) 엔드포인트에서는 한 질문의 반복 실행 결과가 **하나의
컨텍스트 안에**(`metadata.repeated_outputs` / `metadata.repeated_sql`) 담기며, 컨텍스트 N개는 서로 다른
질문 그룹 N개를 뜻한다:

```bash
curl -s -X POST http://127.0.0.1:8000/rag/recall_at_k \
  -H 'Content-Type: application/json' \
  -d '{
        "contexts": [
          {"metadata": {"retrieved_ids": ["d1","d4","d9"], "relevant_ids": ["d1","d4"]}},
          {"metadata": {"retrieved_ids": ["d8","d9"],      "relevant_ids": ["d1","d4"]}}
        ],
        "params": {"k": 5}
      }'
```

```jsonc
{
  "family": "rag", "metric": "recall_at_k", "n": 2, "n_errors": 0,
  "results": [
    {"metric": "recall_at_k", "score": 1.0, "passed": null, "confidence": null,
     "cost": {"tokens": 0, "usd": 0.0, "latency_ms": 0.05}, "detail": {}, "error": null},
    {"metric": "recall_at_k", "score": 0.0, "passed": null, "confidence": null,
     "cost": {"tokens": 0, "usd": 0.0, "latency_ms": 0.03}, "detail": {}, "error": null}
  ],
  "aggregate": {"metric": "recall_at_k", "value": 0.5, "ci_low": 0.0, "ci_high": 1.0,
                "n": 2, "n_errors": 0, "aggregation": "mean", "higher_is_better": true}
}
```

집계는 라이브러리의 `evaluate()`가 그대로 계산한다 — 이진(RATE) 메트릭은 Wilson 구간을 적용한
통과율이고, 등급형(MEAN) 메트릭은 군집 강건(cluster-robust) 신뢰구간을 적용한 평균이다(서로 독립이
아닌(non-iid) 항목에는 `metadata.cluster_id`를 설정한다). 모든 항목은 정확히 한 번씩만 채점되며,
judge 엔드포인트는 전송된 컨텍스트마다 judge를 한 번씩만 실행한다(설정된 judge당 LLM 호출 1회 —
llm_judge의 기준별 기본 동작은 그 호출이 **기준마다** 일어나므로 컨텍스트당 5회이고, PoLL 패널이면
패널 크기만큼 배수가 된다). `n`은 전송된 컨텍스트 수를 세고, `aggregate.n`은
실제로 채점된 항목만 센다(오류가 난 항목은 분모에서 제외된다). 메트릭이 기준별 점수를 보고하면
(llm_judge의 기본 동작) 각 항목의 `detail.criteria`에 그 점수가 담기고, 집계에는 기준별 평균이 담긴
`breakdown` 필드가 추가된다.

## 오류 모델

| 조건 | 결과 |
|---|---|
| 항목을 채점할 수 없음(필드/메타데이터 누락) | **200**; 해당 항목의 `results[i].error`가 설정되고, 그 항목은 집계의 분모에서 제외된다(`aggregate.n`은 유효한 항목만 센다). 오류가 난 항목 ≠ 0점을 받은 항목. |
| 잘못된 형식의 본문, 비어 있는 `contexts`, 알 수 없거나 잘못된 `params` | 상세 정보가 담긴 **422** |
| 메트릭의 선택적 의존성이 서버에 없음 | 정확한 `pip install 'agent-eval[...]'` 힌트가 담긴 **501** |

## Judge 설정(환경 변수, 서버 시작 시)

| 변수 | 의미 |
|---|---|
| `AGENT_EVAL_JUDGE_FACTORY` | `module:attr` 또는 `path/to/file.py:attr` — YAML의 `judge.factory` 규약 그대로다. 백엔드, 인자 없는 콜러블, 또는 리스트(= PoLL 심판 패널)를 반환할 수 있다. 다른 모든 설정에 우선한다. |
| `AGENT_EVAL_JUDGE_PROVIDER` | `_BASE_URL`을 직접 지정하지 않았을 때 쓰는 프리셋 전환 스위치: `anthropic`(Anthropic의 OpenAI 호환 엔드포인트를 통한 Claude, `ANTHROPIC_API_KEY` 필요) 또는 `openai`(`OPENAI_API_KEY` 필요). `_MODEL`은 프리셋의 기본 모델을 덮어쓴다. |
| `AGENT_EVAL_JUDGE_BASE_URL` | OpenAI 호환 베이스 URL(예: `http://vllm.internal:8000/v1`). `_MODEL`이 함께 필요하다. |
| `AGENT_EVAL_JUDGE_MODEL` | `{base_url}/chat/completions`로 보내는 모델 이름. |
| `AGENT_EVAL_JUDGE_API_KEY` | 선택적인 `Authorization: Bearer` 토큰. |
| `AGENT_EVAL_JUDGE_TIMEOUT` | judge HTTP 타임아웃(초 단위, 기본값 60). |
| `AGENT_EVAL_JUDGE_SYSTEM_PROMPT` | 서버 전역 judge 페르소나 서문(openai 호환 judge에만 적용; 요청의 `system_prompt` 파라미터가 이를 재정의한다). 설정되면 `GET /health`의 judge 정보에 "custom system prompt"가 표시된다. |

아무것도 설정하지 않으면 → 결정론적 어휘 중첩(lexical-overlap) **스텁**으로 동작한다(오프라인이며,
실제 판정이 아니다). 어떤 judge가 활성인지는 `GET /health`가 보고한다 — judge 기반 점수를 신뢰하기
전에 먼저 확인한다.

## 운영 참고 사항

- 인증/TLS가 없다 — 폐쇄망(closed network)을 전제로 만들어졌으므로, 외부 노출이 문제가 된다면
  게이트웨이를 앞단에 둔다.
- 배치 크기 상한이 없다: 메모리와 지연 시간은 `len(contexts)`에 비례해 커지고, judge 엔드포인트는
  항목마다 설정된 judge 수만큼 LLM을 호출한다(PoLL 패널이면 그만큼 배수가 되고, llm_judge의 기준별
  기본 동작은 기준 수만큼 — 항목당 5회 — 배수가 되며, 일관성 엔드포인트는
  실행 쌍 N(N−1)/2만큼 추가로 배수가 된다). `--workers`로
  스케일 아웃한다(채점은 요청 단위로 동기 실행된다).
- `uvicorn[standard]`는 컴파일된 휠(wheel)인 `uvloop`, `httptools`, `watchfiles`, `websockets`를
  함께 설치한다. 내부 미러가 사용 중인 플랫폼용 휠을 제공하지 못한다면 플레인 `uvicorn`을 대신
  설치한다 — 서버는 그것만으로도 문제없이 동작한다.

## 폐쇄망 이미지 빌드

인터넷이 연결된 머신에서 휠하우스(wheelhouse)를 미리 만들어 빌드 컨텍스트에 넣고, pip이 그것을
바라보게 한다:

```bash
# connected machine — resolve the full closure for linux/py3.12
docker run --rm -v "$PWD:/w" -w /w python:3.12-slim \
  pip download ".[server,t2s]" -d wheelhouse/

# closed network — add `COPY wheelhouse /wheels` above the pip install line in the
# Dockerfile, then:
docker build --build-arg PIP_ARGS="--no-index --find-links=/wheels" -t agent-eval-api .
```

또는 `PIP_ARGS`가 내부 미러를 가리키게 한다(`--index-url https://pypi.internal/simple`).
**`PIP_ARGS`에 자격 증명을 절대 넣지 않는다** — 빌드 인자는 이미지 히스토리에 기록되기
때문이다(`docker history`). 인증이 필요 없는 미러를 쓰거나, `pip.conf`는 BuildKit 시크릿
마운트로 전달한다.
