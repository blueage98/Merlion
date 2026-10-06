---
name: make_notes
description: Use at the end of a work session (e.g. "세션 종료", "마무리", "다음 세션 준비", "make notes") to write a handoff note, SESSION_NOTES.md at the repo root, that tells the next session what to do next, what state the repo is in (uncommitted changes, running servers, environment), and what was decided and why. Also makes sure CLAUDE.md points the next session at that note. Does not commit, stop servers, or run long test suites on its own.
---

# Make Session Notes

Purpose: leave a handoff note so the next session can pick up the work without re-deriving context. The note lives at `SESSION_NOTES.md` in the repo root and is written in Korean (the user's language). The next session finds it through a one-line pointer in `CLAUDE.md`.

## Steps

1. **Read the previous note** (`SESSION_NOTES.md`) if it exists. Keep any "다음 세션 작업" item that wasn't finished in this session, and drop the ones that were.

2. **Gather the facts. Check them, don't recall them from memory:**
   - `git status --short` and `git diff --stat`: uncommitted changes, grouped by feature/topic.
   - `git log --oneline -5`: the last commits.
   - Running dashboard servers: `netstat -ano | grep -E ":805[0-9] " | grep LISTEN` (Windows) or `lsof -i :8050` elsewhere. For each one, note the port, PID, and which checkout/venv it serves (e.g. `merlion_original/.venv312` on 8051).
   - Environment notes that matter for the next session, e.g. the interpreter, pinned versions (dash 2.18.2, numpy<2.0), and any extra venvs/clones such as `../merlion_original`.
   - Test results: report the last results actually seen in this session, along with the command that produced them. Do not run the full test suite just for the note. If no tests were run this session, say so.

3. **Work out the next tasks.** Take them, in this order of priority, from:
   1. what the user explicitly said they want to do next (quote the intent faithfully);
   2. unfinished items carried over from the previous note;
   3. open issues found this session that the user deferred, or hasn't decided on yet. Mark these as "결정 필요" rather than as a task.

   If the user hasn't said what comes next and nothing carries over, ask them (one short question) before writing.

   For each task, add a concrete starting point: files/functions to look at, an existing pattern to follow (e.g. `ForecastModel.recommend_lgbm_params()` in `merlion/dashboard/models/forecast.py`), and how to verify (which tests, which dashboard flow).

4. **Write `SESSION_NOTES.md`**, overwriting the previous version, with this structure:

   ```markdown
   # 세션 노트 (마지막 업데이트: YYYY-MM-DD)

   ## 다음 세션 작업
   1. <task> — 시작점: <files/functions>, 참고 패턴: <...>, 검증: <tests/flow>
   ...

   ## 결정 필요
   - <open question>: <options, with a recommendation if there is one>

   ## 현재 상태
   - 커밋되지 않은 변경: <grouped by topic, with files>
   - 실행 중인 서버: <port / PID / checkout> (없으면 "없음")
   - 환경: <interpreter, venvs, pinned versions>
   - 최근 테스트 결과: <command → result>

   ## 이번 세션에서 한 일
   - <topic>: <what changed and why, 1–2 lines each>

   ## 알게 된 사실 / 주의사항
   - <non-obvious findings the next session would otherwise rediscover, e.g. root causes, known pre-existing failures>
   ```

   Use absolute dates (`2026-10-05`), not "today"/"yesterday". Keep each bullet to one or two lines. The point is to hand off, not to keep a full log; git history already records the details.

5. **Make sure `CLAUDE.md` points at the note.** If `CLAUDE.md` doesn't already mention `SESSION_NOTES.md`, add this section once, at the end:

   ```markdown
   ## Session Handoff

   If `SESSION_NOTES.md` exists at the repo root, read it at the start of a session: it lists the next tasks, open decisions, and the current repo state left by the previous session (written by the `make_notes` skill).
   ```

6. **Report to the user** in Korean: the path of the note, the list of next tasks, and anything still open at session end. Running servers and uncommitted changes go here too. **Ask** whether to stop the servers or commit. Don't do either on your own.

## Not this skill's job

- Don't commit, push, or stash. Uncommitted work is recorded in the note, and committing it is the user's decision.
- Don't stop running servers. List them and ask.
- Don't run long test suites or benchmarks just to fill in the note.
- Don't store the note in Claude's memory directory. It belongs in the repo, so the user (and teammates) can read and edit it.
