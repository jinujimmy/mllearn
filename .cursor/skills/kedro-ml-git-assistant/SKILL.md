---
name: kedro-ml-git-assistant
description: Saves Kedro/ML lesson work to GitHub (commit + push) on the current feature branch. Use when the user invokes /kedro-ml-git-assistant, asks to publish/push/update GitHub, or when a notebook or Kedro step is finished and the next step is about to start.
---

# Kedro ML Git assistant

Host is **GitHub** (`gh`), not Cursor Origin / share. Do not create a Cursor-hosted remote if `origin` already points at GitHub.

## When to run

- User says `/kedro-ml-git-assistant`, publish, push, or update GitHub.
- A modelling or Kedro **step just succeeded** and the user wants to go to the **next** step — save first, then continue teaching.

Invoking this skill is permission to commit and push the relevant lesson files.

## Do not commit

- `data/**` CSVs (gitignore)
- `conf/**/*credentials*`
- `.venv/`
- Secrets, `.env`

Do commit: `src/`, `conf/base/` (catalog + parameters), `tests/`, `notebooks/*.ipynb`, `.cursor/skills/`, `notes/later-optional.md` when those changed.

## Steps

1. `git status -sb`, `git diff`, `git log -5 --oneline`, `git remote -v`.
2. Stay on the current feature branch. If still on `main` after a new lesson, create `feature/<short-name>` first.
3. Stage only lesson files (not ignored data).
4. Commit with a 1–2 sentence message (why, not file list). HEREDOC. No `--amend` unless the user asked and the amend rules allow it.
5. `~/.local/bin/gh auth setup-git` if `git push` asks for a username.
6. `git push -u origin HEAD` if the branch has no upstream; otherwise `git push`.
7. Reply with the GitHub branch URL. Do not open a PR unless the user asked.

## After a failed push

If `gh` is missing or logged out, install/login as before (device flow), then retry push. Do not switch the remote to Cursor Origin.
