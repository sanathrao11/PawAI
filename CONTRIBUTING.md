# Contributing

## Branching Strategy

```
main        → production (auto-deployed)
develop     → integration branch
feature/*   → new features
fix/*       → bug fixes
```

## Workflow

### 1. Branch from develop

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<feature-name>
```

### 2. Define scope before coding

Before writing code, define what the feature includes and explicitly what it does not include. Keep branches focused — one capability per PR.

### 3. Commit conventions

```
feat: add websocket updates
fix: correct REDIS_URL missing scheme prefix
test: add queue task integration tests
docs: update API reference with new endpoint
chore: bump celery to 5.4
```

### 4. Push and open PR

```bash
git push origin feature/<feature-name>
```

Open a pull request targeting `develop`.

**PR description template:**

```markdown
## What
Brief description of what this PR does.

## Why
Why this change is needed.

## Changes
- `file.py`: reason
- `other.py`: reason

## How to test
1. Step one
2. Step two

## Checklist
- [ ] Tests added or updated
- [ ] Lint passes (`ruff check app tests`)
- [ ] Coverage ≥ 70% (`pytest --cov=app --cov-fail-under=70`)
- [ ] No breaking changes to existing endpoints
```

### 5. Merge to develop, then main

- PR reviewed and approved → merge into `develop`
- When `develop` is stable → merge into `main`
- Merging to `main` triggers automatic deployment to AWS ECS

## Running Tests Locally

```bash
pip install -r requirements.txt
pytest tests/ --cov=app --cov-report=term-missing
```

## Code Style

Linting enforced via `ruff`. Run before pushing:

```bash
ruff check app tests
```

Fix auto-fixable issues:

```bash
ruff check app tests --fix
```
