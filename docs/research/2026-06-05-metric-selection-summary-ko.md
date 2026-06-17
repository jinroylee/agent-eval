# 에이전트 타입별 평가 메트릭 선정 요약 (Metric Selection Summary)

- **작성일**: 2026-06-05
- **근거 문서**: [`2026-06-05-agent-eval-sota-research.md`](./2026-06-05-agent-eval-sota-research.md)
- **대상 코드**: [`src/agent_eval/metrics/`](../../src/agent_eval/metrics), [`src/agent_eval/agents/`](../../src/agent_eval/agents)
- **목적**: 4개 에이전트 타입별로 *어떤 메트릭을 / 왜 / 어떤 데이터 필드로* 측정하기로 결정했는지 1–2p로 정리. 마지막에 **리포지토리에 이미 구현된 메트릭 목록**을 첨부.

본 프레임워크는 4개 에이전트 타입(**Plain · RAG · Orchestration · T2S**)과, 모든 타입에 공통으로 얹히는 **Cross-cutting 레이어(Judge · Performance · Statistics/Governance)** 로 구성된다. 각 타입은 *오프라인 평가용 suite registry* 와 *런타임 크리틱(critic)* 을 각각 가진다.

> **갱신 (2026-06-17): 구현 메트릭 세트가 타깃 세트로 축소됨.** **제거된 메트릭:** 결정론 베이스(`exact_match`·`regex_match`·`set_match`·`numeric_tolerance`·`json_shape`), 검색 보조(`hit_at_k`·`mrr`·`ndcg_at_k`), RAGAS 모드(`answer_relevancy`·`context_recall`). 아래 **§2 근거표는 2026-06-05 당시의 *선정 근거*를 보존**하며(연구 맥락), **현재 실제 구현 현황은 §5와 [`agent-types.md`](../agent-types.md) 카탈로그**가 기준이다.

---

## 1. 설계 원칙 (메트릭 선택의 4가지 근거)

| 원칙 | 내용 | 근거 (출처) |
|---|---|---|
| **객관식 → 결정론적 검증자** | 정답이 객관적으로 검증 가능한 작업(SQL 실행, tool-call 인자, 검색 ID)은 **LLM judge로 게이트하지 않는다**. 실행/AST/집합 비교로 판정. | JudgeBench: GPT-4o judge도 객관 과제에서 랜덤 수준 (Tan et al., ICLR 2025). 자기검증은 추론을 *붕괴*시키고 외부 검증자만이 회복 (Stechly et al., 2024) |
| **주관식 → LLM-as-judge + 패널** | 정답이 없는 품질(일관성·유용성·지시 준수)만 LLM judge가 채점. 고위험 게이트는 다양한 모델 패밀리의 **PoLL 패널**로. | G-Eval (Liu et al., EMNLP 2023); PoLL은 단일 GPT-4 judge보다 인간 상관 ↑·비용 ~7배↓ (Verga et al., 2024) |
| **런타임은 cheap→strong 티어** | 크리틱은 `결정론 → 불확실성(uncertainty) → judge` 순으로 escalate. 신뢰도 임계값 미달 시에만 재시도/상위 모델 호출. | Trust-or-Escalate: 신뢰도 게이팅으로 통계적 인간 일치 보장 (Jung et al., ICLR 2025) |
| **신뢰성·통계적 게이팅** | 단일 pass@1이 아니라 **pass^k**(k회 전부 성공)로 신뢰성 측정. 작은 골든셋엔 CLT 금지, Wilson/exact CI 사용. | tau-bench/tau2-bench pass^k (Yao et al., 2024); 소표본 CLT 금지 (Bowyer et al., ICML 2025) |

---

## 2. 에이전트 타입별 메트릭

> 표기: **(O)** 오프라인 게이트/진단, **(R)** 런타임 크리틱. *필요 데이터 필드*는 `EvalContext`(§4)의 필드명.

### 2.1 Plain (단일 호출 에이전트)

추적할 retrieval/trajectory가 없어 **응답 품질 judge + 라벨-프리 불확실성**이 핵심.

| 메트릭 | 모드 | 선정 근거 (출처) | 필요 데이터 필드 |
|---|---|---|---|
| `selfcheck_consistency` (SelfCheckGPT) | R | 오라클 없이 샘플 일관성으로 환각 탐지 (Manakul et al., EMNLP 2023) | `output`, `metadata.samples` (사전 샘플 N개) |
| `semantic_entropy` | R | 의미 클러스터 엔트로피 기반 환각 신호, Nature 검증 (Farquhar et al., 2024) | `output`, `metadata.samples` |
| `judge` (G-Eval/DAG) | O | 주관 품질(coherence·helpfulness·instruction-following)은 judge가 담당 | `input`, `output`, `criteria`(루브릭) |
| deterministic (format/compliance) | O | 형식·컴플라이언스는 결정론적 pass/fail | `output`(+`expected`/패턴) |

### 2.2 RAG (벡터 + 지식그래프 QA)

검색 품질이 답변 품질의 상한이므로 **Recall@k가 하드 게이트**. 생성 단계는 **groundedness/faithfulness**.

| 메트릭 | 모드 | 선정 근거 (출처) | 필요 데이터 필드 |
|---|---|---|---|
| `recall_at_k` | O | 검색 품질이 답변 상한 → **게이트 메트릭** | `retrieved`(`output`/`metadata.retrieved_ids`, 랭킹 리스트) + `gold`(`expected`/`metadata.relevant_ids`) |
| `ndcg_at_k` | O | 등급별 관련도 반영, BEIR 헤드라인 지표 (Järvelin & Kekäläinen, 2002) | 위와 동일 (관련도 라벨) |
| `precision_at_k`, `mrr`, `hit_at_k` | O | 정밀도/첫 정답 순위/적중 여부 진단 | 위와 동일 |
| `retrieval_sufficiency` | R | CRAG식 사전 게이트: Recall<임계 → 재검색/웹 폴백 (Corrective RAG, Yan et al. 2024; cf. Self-RAG, Asai et al. ICLR 2024 Oral) | `retrieved` + `gold` (+ `min_recall`) |
| `ragas_faithfulness` / `answer_relevancy` | O/R | claim 단위 entailment로 환각 탐지, RAG 표준 (Es et al., EACL 2024) | `input`, `output`, `retrieved_context` |
| `ragas_context_precision` / `context_recall` | O | 라벨 없을 때 LLM-judged 검색 정확도 | `input`, `retrieved_context`, `expected` |
| `semantic_entropy` | R | groundedness 보조 신호 (오라클 부재 시) | `output`, `metadata.samples` |

> ⚠️ Faithfulness는 *검색 맥락에 대한 근거성*만 측정하므로, 반드시 검색 품질 메트릭과 **짝지어** 써야 한다(쓰레기에 근거해도 통과하는 것 방지). 고전 IR 지표는 다운스트림 답변 품질을 불완전하게 예측함(Trappolini et al., 2025).

### 2.3 Orchestration (멀티스텝 tool-use / 플래닝)

도구 호출은 객관 검증 대상 → **결정론적 trajectory/AST 매칭으로 게이트**, judge는 보조.

| 메트릭 | 모드 | 선정 근거 (출처) | 필요 데이터 필드 |
|---|---|---|---|
| `trajectory_match` (strict/unordered/**subset**/superset) | O | subset = "금지 도구 미사용" 하드 안전 게이트; LangGraph 표준 매처 | `trajectory`(`[{tool,args}]`), `expected`(기준 trajectory) |
| `tool_arg_validity` | O/R | BFCL식: 알려진 도구·필수 인자·타입 일치 검증 (BFCL, ICML 2025) | `trajectory`, `metadata.tool_schemas`(`{tool:{arg:type}}`) |
| `agentevals_*` (어댑터) | O | LangGraph 메시지 포맷·graph-trajectory 변형이 필요할 때 | `trajectory`, `expected` |
| `judge` (trajectory) | O | plan 품질·효율·정책 준수는 judge가 채점(항상 결정론 체크와 병행, 단독 금지) | `input`, `trajectory`, `criteria` |
| perf (`pass^k`, latency) | O | 멀티에이전트 신뢰성·토큰/지연 귀속 (§3) | `metadata.latency_ms/tokens`, spans |

> 근거: AgentRewardBench — 단일 trajectory judge는 일반화 실패·실패를 과신뢰 → 인간 보정·결정론 병행 필수. Node/Edge F1은 "일부 벤치마크가 채택한" 진단용이지 단독 게이트가 아님.

### 2.4 T2S (Text-to-SQL & Text-to-Cypher)

**실행 기반 검증(execution-grounded)** 이 정석. LLM judge는 쿼리 정확성에 쓰지 않는다.

| 메트릭 | 모드 | 선정 근거 (출처) | 필요 데이터 필드 |
|---|---|---|---|
| `execution_accuracy` (denotation match) | O | 결과셋 동치 — **1차 게이트** (BIRD, NeurIPS 2023) | `output`(예측 SQL), `expected`(정답 SQL), `metadata.db_ref`(픽스처 DB) |
| `soft_f1` | O | 셀-백 F1, 부분 점수 (graded) | `output`, `expected`, `metadata.db_ref` |
| `component_match` | O | tables+projections AST 중첩 — *진단용*(높은 false-negative) | `output`, `expected` (DB 불필요) |
| `ast_valid` | R | 파싱 가능 여부 (런타임 tier-1) | `output` |
| `schema_linking` | R | 참조 테이블/컬럼이 스키마에 실존하는지 | `output`, `metadata.db_ref` |
| `exec_validity` | R | 드라이런 무오류 실행 | `output`, `metadata.db_ref` |
| `judge` (NL-intent only) | O | judge는 **자연어 의도·설명 충실도**에만 한정 (쿼리 정확성 ✗) | `input`, `output`, `criteria` |

> 근거: 결정론적 실행/AST 체크가 곧 T2S의 크리틱 (CRITIC, Gou et al. 2024 — 도구 제거 시 자기수정 이득 붕괴). Test-Suite Accuracy로 EX의 false-positive 보강(Zhong et al., EMNLP 2020). 효율은 R-VES(BIRD 리더보드 최신).

---

## 3. Cross-cutting 레이어 (모든 타입 공통)

| 레이어 | 메트릭 / 함수 | 선정 근거 (출처) | 필요 데이터 필드 |
|---|---|---|---|
| **Judge** | `judge` (단일/PoLL 패널), `poll_pointwise`, `pairwise_swap_average` | 다양한 패밀리 패널이 자기선호 편향 상쇄 (Verga et al. 2024); swap-and-average로 위치 편향 제거 (Zheng et al. NeurIPS 2023) | `input`, `output`, `criteria`, judge backend |
| **Performance** | `latency_budget`, `token_budget`, `attribute`(노드별 귀속), `latency_percentiles`(p50/95/99), `scenario_pass_hat_k`, `cost_of_pass` | 평균이 아닌 **꼬리 지연(p95/99)** 보고 (Revisiting-SLO); 정확도+비용 단일화 Cost-of-Pass (Erol et al., ICLR 2026) | `metadata.latency_ms/tokens`, spans(`{node,tokens,latency_ms,usd}`), `per_task_runs` |
| **Statistics / Governance** | `wilson_interval`·`clopper_pearson`·`beta_binomial_ci`, `ppi_interval`(PPI++), `cohen_kappa`, `clustered_se`, `anytime_valid_sequence`, `mcnemar`·`paired_bootstrap`·`wilcoxon`·`benjamini_hochberg`, `bradley_terry`, `certify_judge`·`judge_gate_ci` | 소표본 exact CI (Bowyer, ICML 2025); judge-인간 일치 인증 후 게이트 연결 (CALM, Ye et al. 2024); 상대 랭킹은 Bradley-Terry (Chatbot Arena, ICML 2024) | 성공/시행 카운트, 인간 라벨, 클러스터 ID, 페어 비교 |

---

## 4. 데이터 스키마 요약 — `EvalContext` (점수 산출에 필요한 입력)

모든 메트릭은 단일 `EvalContext`를 입력받아 비교·채점한다. 필드 ↔ 사용 메트릭 매핑:

| 필드 | 타입 | 주 용도 | 사용 메트릭(예) |
|---|---|---|---|
| `input` | Any | 질의/프롬프트 | judge, ragas |
| `output` | Any | 에이전트 출력(답변/SQL/검색ID) | 거의 전부 |
| `expected` | Any | 정답(골드 SQL·기준 trajectory·gold IDs) | execution_accuracy, trajectory_match, recall_at_k |
| `retrieved_context` | Sequence[str] | 검색된 본문 | ragas_* |
| `trajectory` | Sequence[{tool,args}] | tool-call 시퀀스 | trajectory_match, tool_arg_validity |
| `metadata.retrieved_ids` / `relevant_ids` | list | 랭킹 검색 ID / 골드 관련 ID | recall@k / precision@k |
| `metadata.db_ref` | str | 실행용 픽스처 DB 경로 | execution_accuracy, schema_linking, exec_validity |
| `metadata.tool_schemas` | dict | 도구별 인자·타입 스펙 | tool_arg_validity |
| `metadata.samples` | list | 사전 샘플링된 후보 답변 | selfcheck_consistency, semantic_entropy |
| `metadata.latency_ms` / `tokens` | num | 성능 예산 | latency_budget, token_budget |

> 예) **precision@k** 점수를 내려면 → `retrieved`(랭킹 검색 결과) + `gold`(정답 관련 ID)가 필요. **execution_accuracy** → 예측 SQL + 정답 SQL + 실행 DB(`db_ref`).

---

## 5. 리포지토리 구현 현황 (이미 구현된 메트릭 목록)

✅ = 구현·기본 등록됨 · ⚙️ = 선택적 어댑터(추가 의존성 필요) · 🔧 = 함수형 헬퍼

**`deterministic_.py` (공통, FREE 티어)** ❌ **제거됨 (2026-06-17)** — `default_registry()`는 이제 빈 베이스. 필요 시 `BaseMetric`로 직접 추가([`customizing.md`](../customizing.md)).

**`retrieval_.py` (RAG)** ✅
`recall_at_k` · `precision_at_k` · `retrieval_sufficiency`(CRAG 런타임 게이트)  *(`hit_at_k`·`mrr`·`ndcg_at_k`는 2026-06-17 제거)*

**`trajectory_.py` (Orchestration)** ✅
`trajectory_match`(strict/unordered/subset/superset) · `tool_arg_validity`(BFCL)

**`ast_query_.py` (T2S)** ✅
`execution_accuracy` · `soft_f1` · `component_match` · `ast_valid` · `schema_linking` · `exec_validity`

**`uncertainty_.py` (Plain·RAG 런타임)** ✅
`selfcheck_consistency`(SelfCheckGPT) · `semantic_entropy`

**`perf_.py` (공통 성능)** ✅
`latency_budget` · `token_budget` · `attribute`🔧 · `latency_percentiles`🔧 · `scenario_pass_hat_k`🔧

**`judge_.py` (공통 JUDGE 티어)** ✅ *(judge backend 주입 필요)*
`judge`(단일 backend 또는 PoLL 패널) — `judges/panel.py`의 `poll_pointwise`·`pairwise_swap_average`, `judges/governance.py`의 인증/게이트 연동

**`stats/` (공통 통계)** ✅
`pass_hat_k` · `cost_of_pass` · `wilson_interval` · `clopper_pearson` · `beta_binomial_ci` · `percentile_ci` · `ppi_interval` · `cohen_kappa` · `clustered_se` · `anytime_valid_sequence` · `mcnemar` · `benjamini_hochberg` · `paired_bootstrap` · `wilcoxon` · `bradley_terry`

**`agentevals_.py` (Orchestration)** ⚙️ — `agentevals_*` (LangGraph 네이티브 trajectory 매처, `[agentevals]` extra)
**`ragas_.py` (RAG)** ⚙️ — `ragas_faithfulness` · `ragas_context_precision` (`[ragas]` extra)  *(`answer_relevancy`·`context_recall`는 2026-06-17 제거)*

**타입별 런타임 크리틱(구현됨)**: Plain=`selfcheck_consistency` · RAG=`retrieval_sufficiency`+`semantic_entropy` · Orchestration=`tool_arg_validity` · T2S=`ast_valid`+`schema_linking`+`exec_validity`.

> **미구현/외부 위임(참고)**: RAGChecker claim-F1, Test-Suite Accuracy, R-VES, PSJS(Cypher), HHEM/Lynx 환각 탐지기, Process Reward Model, Bradley-Terry 부트스트랩 리더보드는 연구 문서에 *권장*으로 남아 있으나 코드에는 아직 없음(또는 어댑터 경유).

---

## 6. 참고문헌 (핵심 출처)

1. **G-Eval** — Liu et al., *EMNLP 2023*. SummEval Spearman ~0.514. (확률가중은 Spearman 향상이 아니라 Kendall-τ tie-break 용도) — https://aclanthology.org/2023.emnlp-main.153/
2. **PoLL (LLM Juries)** — Verga et al. (Cohere), 2024. 3모델 패널 > 단일 GPT-4 judge, ~7–8배 저렴 — https://arxiv.org/abs/2404.18796
3. **Trust or Escalate** — Jung, Brahman, Choi, *ICLR 2025*. 신뢰도 게이팅 런타임 크리틱 청사진 — https://arxiv.org/abs/2407.18370
4. **JudgeBench** — Tan et al., *ICLR 2025*. 객관 과제에서 judge 단독 금지 — https://arxiv.org/abs/2410.12784
5. **Self-Verification Limits** — Stechly, Valmeekam, Kambhampati, 2024. 외부 검증자 필요 — https://arxiv.org/abs/2402.08115
6. **CRITIC** — Gou et al., 2024. 도구 기반 verify-then-correct — https://arxiv.org/abs/2305.11738
7. **RAGAS** — Es et al., *EACL 2024*. faithfulness/answer·context-relevance — https://aclanthology.org/2024.eacl-demo.16/
8. **RAGChecker** — Ru et al. (Amazon), *NeurIPS 2024*. 검색/생성 실패 분리 (오프라인 진단)
9. **nDCG** — Järvelin & Kekäläinen, *ACM TOIS 2002*. BEIR 헤드라인 지표
10. **Self-RAG** — Asai et al., *ICLR 2024 Oral* — https://arxiv.org/abs/2310.11511 · **Corrective RAG (CRAG)** — Yan et al., 2024 — https://arxiv.org/abs/2401.15884 (반사적 검색 크리틱)
11. **Text-to-SQL EX (BIRD)** — Li et al., *NeurIPS 2023*; **Test-Suite Accuracy** — Zhong, Yu, Klein, *EMNLP 2020* — https://aclanthology.org/2020.emnlp-main.29/
12. **BFCL (function-calling AST)** — *ICML 2025*. tool-call 평가 표준
13. **tau-bench / tau2-bench (pass^k)** — Yao et al.; Barres et al., 2024 — https://arxiv.org/pdf/2406.12045
14. **Semantic Entropy** — Farquhar et al., *Nature 2024* — https://www.nature.com/articles/s41586-024-07421-0
15. **SelfCheckGPT** — Manakul et al., *EMNLP 2023*
16. **CALM (judge 편향 감사)** — Ye et al., 2024 — https://arxiv.org/html/2410.02736v1
17. **소표본 CLT 금지** — Bowyer et al., *ICML 2025*; **Adding Error Bars to Evals** — Miller (Anthropic), 2024 — https://arxiv.org/abs/2411.00640
18. **Cost-of-Pass** — Erol et al., *ICLR 2026* — https://arxiv.org/abs/2504.13359

> 모든 SOTA 수치는 자체 라벨 데이터로 재검증 후 게이트에 연결할 것(전이 격차 존재). 상세 검증·반론은 근거 문서 §6 참고.
