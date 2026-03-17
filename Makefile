# Use host UID/GID for container user mapping
UID := $(shell id -u)
GID := $(shell id -g)

# Build CPU image via compose profile
build:
	UID=$(UID) GID=$(GID) docker compose --profile cpu build

# Open a shell in the dev container
bash:
	UID=$(UID) GID=$(GID) docker compose run --rm dev bash

# Start JupyterLab
jupyter:
	UID=$(UID) GID=$(GID) docker compose --profile jupyter up jupyter

# Stop all services
down:
	docker compose down
