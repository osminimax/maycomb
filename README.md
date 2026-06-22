# Maycomb — LLM Mockup Server

사람이 LLM을 흉내 내어 Agent Harness 시스템을 테스트하는 OpenAI 호환 목업 서버입니다.
하네스가 `/v1/chat/completions`로 요청을 보내면 요청이 **오퍼레이터 콘솔**에 큐잉되고,
오퍼레이터가 reasoning / content / tool call을 직접 작성해 응답합니다. 모든 요청·드래프트·
전송 chunk는 SQLite WAL 이벤트 로그에 보존되어 데이터셋으로 export할 수 있습니다.

와이어 포맷은 `wire-format-spec-v1.md`(스키마 버전 `wire/1`)를 따릅니다.

## 설치 / 실행

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m maycomb serve          # http://127.0.0.1:8000
```

- 콘솔: `http://127.0.0.1:8000/` (브라우저)
- OpenAI 호환 API: `http://127.0.0.1:8000/v1`
- 설정 파일: `maycomb.example.toml`을 `maycomb.toml`로 복사 (없어도 기본값으로 동작).
  콘솔 ⚙에서 바꾼 값은 SQLite에 저장되어 파일보다 우선합니다.
- 옵션: `maycomb serve --config <toml> --host <h> --port <p> --data-dir <dir>`

하네스 쪽 설정 예 (OpenAI SDK):

```python
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
```

## 오퍼레이터 워크플로

1. 하네스가 요청을 보내면 콘솔 큐에 나타납니다(WebSocket 실시간 알림).
   non-streaming 요청은 응답 제출까지 대기하고, streaming 요청은 첫 chunk 전
   15초 간격 keep-alive 주석(§4.1)으로 연결이 유지됩니다.
2. **대화 탭**에서 전체 메시지(system/developer/user/assistant/tool)를 확인합니다 —
   하네스가 돌려보낸 tool 결과(검색 결과, skills 파일 내용 등)도 여기서 봅니다.
   **도구 탭**에 요청의 tools 정의, **파라미터 탭**에 나머지 필드(비표준 필드는 ⚠ 배지)가 표시됩니다.
3. 작성기에서 reasoning_content / content / tool_calls를 작성합니다.
   tool call의 arguments는 JSON으로 편집하고 와이어에서는 항상 문자열로 직렬화됩니다(§3).
   드래프트는 자동 저장되며 모든 리비전이 `draft_saved` 이벤트로 남습니다.
4. 전송: **paced**(기본 30 tok/s, TTFT 0ms) / **instant** / **LIVE**(타이핑 그대로 중계, §4.5).
   reasoning → content → tool_calls → finish 순서(§4.2)는 콘솔과 서버가 모두 강제합니다.
5. 필요 시 중단·주입(§8, §9.2): 정상 마감(stop/length, partial 기록), 강제 절단(finish 없이
   TCP 종료), 429/500/context 초과 주입, TTFT 지연, N chunk 후 절단.

## 명세 구현 노트

- 응답 `id`는 `chatcmpl-` + base62 24자이며 내부 exchange UUID와 1:1 매핑(§1) —
  `maycomb.ids.exchange_uuid_for()`로 역변환 가능합니다.
- `system_fingerprint: "fp_mockup_v1"` 고정, `model`은 요청 값 반향(목록 외 모델명도 통과).
- 요청 거부는 최소만: `n>1`, `logprobs:true`, `messages` 누락/빈 배열, JSON 파싱 불가(§2.1, §9.1).
  나머지 필드는 수용·표시·원형 보존, 미지 필드는 `_unknown_fields`로 기록.
- `response_format`(json_object/json_schema)은 제출 시 서버가 검증해 차단하며,
  "강제 우회" 시 `validation_bypassed` + 자동 태그가 기록됩니다. 깨진 tool arguments도 동일.
- `tool_choice` 제약은 경고만 하고 강제하지 않습니다(위반 응답 제작 가능).
- usage는 tiktoken(`o200k_base` 기본, `cl100k_base`/`approx` 전환) **근사치**입니다(§7):
  메시지당 +3, 프라이밍 +3. tools 정의는 prompt 집계에 포함하지 않습니다.
  tiktoken 인코딩 로드가 불가능하면(오프라인 등) 자동으로 approx로 폴백합니다.
  원문이 전부 저장되므로 사후 재계산이 가능합니다.
- streaming의 HTTP 오류 주입(429/500/400)은 **SSE 헤더 전송 전**에만 가능합니다.
  서버는 첫 frame 또는 첫 keep-alive 시점까지 헤더를 지연하므로 기본 15초의 주입 윈도우가
  있으며, **지연 주입을 먼저 걸면** keep-alive와 헤더가 보류되어 윈도우가 무한정 늘어납니다.
- 강제 절단은 ASGI 예외(`OperatorHardAbort`)로 연결을 끊습니다 — 서버 로그에 해당
  traceback이 남는 것은 의도된 동작이며, 클라이언트는 `RemoteProtocolError`류를 관측합니다.
- keep-alive 주석은 chunk가 아니므로 `chunk_sent`로 기록하지 않습니다. `[DONE]`은 기록합니다.
- 서버 재시작 시 살아있던 pending/active 교환은 `exchange_aborted(server_restart)`로 마감됩니다.
- streaming 재구성본(`response_submitted.response`)은 alias 토글과 무관하게
  `reasoning_content` 키로만 적재됩니다(와이어 delta에는 별칭이 함께 나갑니다).

## 저장 구조 (spec §11)

- `data/maycomb.db` — SQLite(WAL 모드): `exchanges`(인덱스), `wal_events`(이벤트 로그),
  `drafts`(최신 드래프트), `settings`(런타임 오버라이드)
- `data/raw/<exchange_id>.json` — 수신 원문 바이트(sha256과 함께 `request_received`에 기록)
- 이벤트 타입: `request_received`, `request_parsed`, `draft_saved`, `response_submitted`,
  `chunk_sent`, `exchange_aborted`, `client_disconnected`, `error_injected` +
  명세 외 확장으로 자동 거부 응답을 기록하는 `request_rejected`
- 모든 이벤트 payload에 `v: "wire/1"` 포함. non-streaming `response_submitted.response_raw`는
  클라이언트 수신 바이트와 동일하며, streaming은 chunk 전수 기록 + 재구성본 이중 표현.

## CLI

```powershell
maycomb serve   [--config maycomb.toml] [--host] [--port] [--data-dir]
maycomb export  [--status completed|aborted|injected|rejected|all]
                [--exclude-partial] [--strip-reasoning] [--out export.jsonl]
maycomb verify  # chunk_sent 재생 == response_submitted 재구성본 대조 (§11 검증 잡)
```

## 콘솔 API (내부)

`GET /api/state` · `PUT /api/config` · `GET /api/exchanges[?status=]` ·
`GET /api/exchanges/{id}` / `/events` / `/raw` ·
`POST /api/exchanges/{id}/draft` / `/validate` / `/submit` / `/abort` / `/inject` ·
`WS /api/ws` (실시간 알림 + LIVE 모드 중계)

콘솔 API에는 인증이 없으므로 기본 바인딩(127.0.0.1)을 유지하세요.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest            # 43 tests
.\.venv\Scripts\python.exe scripts\selftest.py  # 실서버 종단간 (serve 후 실행)
.\.venv\Scripts\python.exe scripts\demo_client.py  # 수동 데모 (콘솔에서 직접 응답)
```
