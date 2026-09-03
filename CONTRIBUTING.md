# Contributing

Thanks for contributing to PhysAI Robot Starter. Keep changes small, focused,
and compatible with the module boundaries in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development workflow

1. Start from an up-to-date branch and create a focused feature, fix, docs, or
   test branch.
2. Read the relevant implementation and tests before editing.
3. Keep robot, task, planner, policy, and infrastructure concerns in their
   owning modules.
4. Add or update tests for behavior changes.
5. Run the focused test first, then the full suite for cross-module changes.
6. Review the final diff for unrelated files, generated artifacts, and secrets.

## Commit messages

Use one purpose per commit with this format:

```text
[type] imperative summary
```

Supported types:

- `[feat]` for a new capability or modular boundary.
- `[fix]` for a behavior correction.
- `[docs]` for documentation-only changes.
- `[test]` for test-only changes.
- `[refactor]` for internal restructuring without behavior changes.
- `[perf]` for a performance improvement.
- `[ci]` for continuous integration changes.
- `[build]` for packaging or build changes.
- `[chore]` for maintenance that does not change behavior.
- `[revert]` for reverting an earlier commit.

Examples:

```text
[feat] add policy registry
[fix] handle missing camera frame
[docs] clarify local model setup
[test] cover robot capability checks
```

Keep the summary concise, use an imperative verb, and omit a trailing period.
Do not combine unrelated code, documentation, and roadmap changes in one
commit. A roadmap update should normally be a separate `[docs]` commit.

## Validation

Install development dependencies and run the test suite from the project root:

```bash
uv sync --extra dev
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run python -m pytest tests/ -q
```

Use a focused test path when iterating. For documentation-only changes, check
local links and command names against the current repository.

## Repository hygiene

Do not commit downloaded assets, model snapshots, demonstrations, videos,
generated plans, virtual environments, or credentials. These belong in the
ignored local directories documented in [README.md](README.md).

Before opening a pull request, confirm that:

- the commit messages follow the format above;
- tests and relevant validation pass;
- documentation matches the current behavior;
- the diff contains no secrets or generated artifacts; and
- the change preserves the ownership rules in the architecture document.