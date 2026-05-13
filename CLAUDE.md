# CLAUDE.md — TamuBot

RAG chatbot for Texas A&M course/policy info. Module-level detail: `src/tamubot/rag/CLAUDE.md`, `src/tamubot/ingestion/CLAUDE.md`, `src/tamubot/evals/CLAUDE.md`

## Environment

Claude Code runs **inside** Docker container `tamubot-dev-1`. No Docker-in-Docker. Python packages installed container-wide (no `.venv`).

## Commands

```bash
streamlit run src/tamubot/app/streamlit.py --server.headless true  # start app (port 8501)
make test | lint | typecheck | format | probe | probe-full
```

## Gotchas

- **Config**: always `from tamubot.core import config` — never `os.getenv()` directly.
- **Skills**: discovery uses `~/.claude/skills/<name>/SKILL.md`, not project-level `.claude/skills/*.md`. If a skill doesn't appear, check for broken symlinks — fix from Windows PowerShell, not inside the container.

## LLM API usage
If not specifically asked, dont use api calls which exceed 10 cals, ask if unsure