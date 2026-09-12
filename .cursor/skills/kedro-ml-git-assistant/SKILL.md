---
name: kedro-ml-git-assistant
description: Saves one Kedro/ML lesson process to GitHub (commit + push). Use when the user invokes /kedro-ml-git-assistant, asks to publish that step, or has finished learning one process (one node plus its YAML) and is ready before the next process starts.
---

# Kedro ML Git assistant

Host is **GitHub** (`gh`), not Cursor Origin / share. Do not create a Cursor-hosted remote if `origin` already points at GitHub.

## One process per save

Do **not** commit the whole pipeline at once. One save = one process the user just learned, for example:

- `prepare_model_table` in `nodes.py` + `leakage_or_id` in `parameters.yml` + `model_table` in `catalog.yml` + the matching `node(...)` in `training.py`

Do **not** include the next function (`time_split`, train, evaluate, lags) in the same commit.

Order for each process:

1. Change `nodes.py` for that function only (or the smallest set needed).
2. Change `parameters.yml` / `catalog.yml` / pipeline wiring **only if this process needs them**.
3. User learns it.
4. This skill: commit those files and push.
5. Only then start the next process.

Invoking this skill is permission to commit and push **that process**, not later work sitting in the working tree. If extra files belong to the next step, leave them unstaged and say so.

## When to run

- User says `/kedro-ml-git-assistant`, or publish/push **this step**.
- Teaching is about to move to the **next** process — save the current one first.

## Do not commit

- `data/**` CSVs
- `conf/**/*credentials*`
- `.venv/`
- Secrets, `.env`

## Steps

1. `git status -sb`, `git diff`, `git log -5 --oneline`, `git remote -v`.
2. Stay on the current feature branch (or `feature/<process-name>` if still on `main`).
3. Stage **only** this process’s files.
4. Commit with a 1–2 sentence message naming the process. HEREDOC.
5. `~/.local/bin/gh auth setup-git` if push asks for a username.
6. `git push -u origin HEAD` if no upstream; else `git push`.
7. Reply with the branch URL. No PR unless asked.

## After a failed push

Fix `gh` auth; retry push. Do not switch the remote to Cursor Origin.
