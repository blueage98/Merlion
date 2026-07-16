---
name: analyze-codebase
description: Use when the user wants a structured architecture analysis of a repository (넓게/좁게/흐름 style) - runs six steps (stack/deps, folder & layer structure, core data models, endpoints & status/error signaling, one request trace, hidden conventions & weaknesses) and produces a CLAUDE.md-ready summary plus a weakness list. Input is a repo root path; output is a summary to merge into CLAUDE.md and a list of discovered weaknesses.
---

# Analyze Codebase

Six-step architecture analysis of a repository. Input: a repo root path (default: current working directory). Output: (1) a CLAUDE.md-ready summary section, (2) a list of discovered weaknesses/hidden conventions.

Do NOT guess. Every claim must be backed by an actual file path (and line number where relevant) found via Glob/Grep/Read or an Explore/general-purpose subagent. If a step doesn't apply to this codebase (e.g. no HTTP layer), say so explicitly instead of forcing an answer.

For steps that only require reading a handful of files, do them directly. For steps that require scanning many files, delegate to the Explore agent (or general-purpose for open-ended digging) with a self-contained prompt — it has no memory of this skill, so give it full context each time.

## Step 1 — 의존성과 스택 확인 (Dependencies & stack)

Goal: identify language(s), package manager, frameworks, DB/queue/cache tech, and how the project is installed/run.

Prompt template (use directly, or hand to an agent):
```
Repo root: {ROOT}. Identify the technology stack:
- Language(s) and version constraints (check pyproject.toml/setup.py/package.json/go.mod/Gemfile/pom.xml/etc.)
- Package manager and how to install for dev (look for README/CLAUDE.md install instructions)
- Web framework / RPC framework if any (Flask, Express, FastAPI, gRPC, Dash, etc.)
- Database / cache / queue dependencies (check for sqlalchemy, psycopg2, redis, kafka clients, ORMs, etc. in dependency files — do not assume a DB exists just because the task mentions one)
- Test framework and how tests are run
Report each finding with the exact file and line/section it came from. If a category doesn't exist in this repo, say "none found" rather than guessing.
```

## Step 2 — 폴더 구조와 계층 파악 (Folder structure & layers)

Goal: map top-level folders, then locate anything resembling routes/controllers/services/db layers — but only if that pattern actually exists. Many codebases (libraries, CLIs, data pipelines) have no such layering; report that honestly instead of forcing a fit.

Prompt template:
```
Repo root: {ROOT}. List the top-level folder structure (1-2 levels deep) and give a one-line purpose per folder based on actual file contents (not assumptions from names alone).
Then check whether a routes/controllers/services/db (or equivalent MVC-style) layering exists anywhere in the repo — including in sub-packages, GUIs, or admin panels, not just a top-level "api/" folder. For each layer that does exist, list the concrete files/folders that implement it.
If no such layering exists, state that clearly and describe what structure the codebase actually uses instead (e.g. library modules, pipeline stages, CLI commands).
```

## Step 3 — 핵심 데이터 모델 지목 (Core data models / domain vocabulary)

Goal: extract the domain vocabulary — key classes/config objects and what they mean — from actual code.

Prompt template:
```
Repo root: {ROOT}. Based on class names, function names, and config objects actually defined in the code (not documentation), list the key domain classes/concepts. For each: file path, one-line description of its responsibility, and its relationship to other domain classes (e.g. "X wraps Y", "X is persisted via Y"). Prioritize classes that appear across multiple modules or are named in factory/registry patterns.
```

## Step 4 — 주요 엔드포인트와 상태 코드 정리 (Endpoints & success/failure signaling)

Goal: find every externally-triggered entry point (HTTP route, RPC method, CLI command, GUI callback, message-queue consumer) and how it signals success/failure to its caller. If there are no literal HTTP status codes (e.g. a GUI framework like Dash/Streamlit, or a CLI), find the equivalent signaling mechanism instead of skipping the step.

Prompt template:
```
Repo root: {ROOT}. Enumerate all externally-triggered entry points (HTTP routes, RPC handlers, CLI commands, GUI event callbacks, queue consumers — whichever applies here). For each: file:line, trigger/input, and what it returns on success vs failure.
Specifically check: are errors caught (try/except) or do they propagate raw to the caller/user? Is any sensitive info (stack traces, file paths, internals) exposed in error responses? If literal HTTP status codes don't apply, identify the actual success/failure signaling mechanism (exception, error modal/div, exit code, log line, PreventUpdate-style no-op, etc.) and cite where it happens.
```

## Step 5 — 한 요청의 흐름 끝까지 추적 (Trace one request end-to-end)

Goal: pick the single most representative "write" flow (a state-changing action — form submit, POST, CLI command that persists something) and trace it file-by-file from entry point to persistence.

Prompt template:
```
Repo root: {ROOT}. Trace one concrete, representative state-changing flow from its entry point (HTTP handler / CLI command / GUI callback) all the way to wherever data is persisted (DB write, file write, cache write, etc.). List the exact chain of file:function call sites, in order, with a one-line note per hop on what it does to the data. Prefer a flow that touches the layers identified in step 2 (or the closest analog) so the trace validates that structure.
```

## Step 6 — 숨은 규약과 약점 발견 (Hidden conventions & weaknesses)

Goal: surface undocumented conventions and concrete weaknesses — not generic security advice, only things actually observed in this code.

Prompt template:
```
Repo root: {ROOT}. Look for, with file:line evidence for each:
- Implicit conventions the code silently relies on (naming, file-format assumptions, init-order dependencies) that aren't documented anywhere.
- Concurrency/shared-state issues: singletons, module-level globals, or shared directories/caches that could cause cross-request or cross-user bugs.
- Error-handling gaps: places where exceptions are swallowed silently, or where raw tracebacks/internal details leak to callers/users.
- Concrete security concerns actually present in this code: unvalidated file paths (traversal risk), unrestricted uploads, unsafe deserialization (pickle/yaml.load/etc.), missing input validation at trust boundaries.
- Any global mutable state shared across requests/sessions/users.
Only report what you can point to in the code. Do not include generic best-practice advice with no corresponding evidence.
```

## Output format

After running all six steps, produce two deliverables:

1. **CLAUDE.md-ready summary** — a concise "## Architecture" style section (folder structure, layers if present, domain vocabulary, one traced flow) written for a future Claude Code session, in the same style as the rest of the target CLAUDE.md. Ask the user before appending it — do not auto-write to CLAUDE.md.
2. **Weakness list** — a flat, prioritized list of findings from steps 4 and 6, each with a one-line summary, file:line, and why it matters. No fixes unless asked.

Keep total findings-list length reasonable (favor the most load-bearing/severe items first) rather than exhaustively listing every minor nit.
