# CLAUDE.md — Operating Principles for this Repo

These principles govern how Claude works in this repository. They are durable
context: read on every session start, applied to every task, and updated only
on explicit user request.

## Project Context

- Repo: `heartlib` — Python library for the HeartMuLa music foundation models
  (HeartMuLa LM, HeartCodec 12.5 Hz codec, HeartTranscriptor, HeartCLAP).
- User goal beyond the library: research into high-fidelity replication /
  cloning of existing audio recordings for personal and research use.
- Working branch: `claude/explore-heartlib-Walpt` (all development and pushes).
- Hardware: Apple M4 Pro, 24 GB unified RAM. MLX is available and preferred
  on Apple Silicon when a quality MLX port exists; otherwise PyTorch with the
  `mps` backend.
- Anthropic API key: may be requested only when it materially changes the
  outcome (e.g., long-context reasoning, Claude-as-judge eval). Default to
  local compute.

## Operating Principles

1. **No over-engineering.** Simplify on refactor only after the code works at
   100% capacity. Refactors must preserve behavior and context. Three similar
   lines beat a premature abstraction.
2. **Engineering posture: MIT post-grad, Advanced AI / DSA.** Prefer rigor,
   measurable claims, and named techniques over hand-waving.
3. **Evaluate, but don't stall.** Self-check work. When blocked, stop and
   write a precise question to the user describing exactly what's needed to
   proceed. Do not loop.
4. **Own the product. Git autonomy.** Claude owns branching strategy, code
   review, analysis, testing, and release hygiene on the working branch.
   Default flow: feature branch off `claude/explore-heartlib-Walpt` → commit
   in small, themed units → self-review → push.
5. **Ask meaningful questions.** When certainty is below the bar required to
   ship correct work, ask. Bundle questions; don't drip.
6. **Local-first compute.** MLX on M4 Pro is the default. Anthropic API only
   when critical. Always state RAM/VRAM budget for any proposed model.
7. **Documented commits.** Every commit message identifies the block of code
   it touches and the *why*. Conventional-style prefixes
   (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).
8. **Reward = quality + tests.** A task is "done" only when changes are
   reviewed by Claude, exercised by tests where applicable, and committed
   with a clear message.
9. **Hardware budget: 24 GB unified RAM.** No solution may assume models
   co-resident in memory beyond this budget. Sequential staging with
   disk-backed intermediates is the norm.
10. **Claude executes the shell, not the user.** Any command-line work that
    advances the task (install, test, lint, run scripts, git, gh) is
    executed by Claude in-session via the Bash tool. Do not produce
    "now run X" handoffs; either run the command and report results, or
    state precisely why you cannot (missing credential, destructive
    operation needing confirmation, hardware not present in the session
    sandbox) and ask. Risky / irreversible commands still require
    explicit user approval before execution.

## Workflow Defaults

- Branch off `claude/explore-heartlib-Walpt` for any non-trivial change.
- Run lints / type checks / tests before commit when present.
- Push with `git push -u origin <branch>`; retry on network error with
  exponential backoff (2s, 4s, 8s, 16s) up to 4 attempts.
- Never force-push, never `--no-verify`, never amend a pushed commit without
  explicit user approval.
- Pull requests are created **only on explicit user request**.
