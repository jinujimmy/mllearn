---
name: todo
description: Captures work marked later or optional into notes/later-optional.md. Use when the user or agent says something will be done later, is optional, leftover, skipped for now, or deferred; when the user invokes /todo; or when planning next steps that are not in the current task.
---

# Later / optional list

whatever we say will be done later or optional should be noted in this list.

## List file

Path (current project): `notes/later-optional.md`

Create the file if it is missing. Use the template in this skill.

## When to write

Add an item whenever anyone says work is:

- later
- optional
- leftover
- skip for now
- when you are ready
- not this task
- deferred / follow-up / nice to have

Do this in the **same turn** as that statement. Do not only mention it in chat.

## How to add

1. Read `notes/later-optional.md`.
2. Skip if the same item already exists (open or done).
3. Append under `## Open`:

```markdown
- [ ] YYYY-MM-DD — short title. Why it was deferred. Pointers (files, branch, metric).
```

4. In chat, one short line: it was added to the list. Do not dump the whole file unless asked.

## How to show or close

- User asks what is later/optional, or invokes `/todo` → show **Open** items only (or the full file if they ask).
- User completes an item → move it to `## Done` with `- [x]` and the date closed.

## Do not

- Treat the current requested task as later.
- Commit or push the list unless the user asked to save to git.
---
