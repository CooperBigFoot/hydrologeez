# pyplate

Python project boilerplate with `uv`, `ruff`, and `pytest`.

## Quick Start

```bash
gh repo create my-project --template hydrosolutions/pyplate --clone --private
cd my-project
bash init.sh my-project
```

## Development

```bash
uv run pytest                  # run tests
uv run ruff check --fix        # lint
uv run ruff format             # format
```

## Adding Dependencies

```bash
uv add <package>               # runtime dependency
uv add --group dev <package>   # dev dependency
```
