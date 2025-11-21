ruff check --target-version=py312 --fix .
black . --exclude=notebooks --exclude=.venv
