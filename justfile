alias l := local
alias ld := local-down

alias bd := build-docker
alias f := frontend

build-docker:
	export GROUP=backend && \
	if ! pdm lock --check -G $GROUP; then \
		echo "Dependencies have changed, updating lock file..."; \
		pdm lock --update-reuse -G $GROUP; \
	else \
		echo "Lock file is up-to-date. Skipping update."; \
	fi; \
	pdm export -G $GROUP -o ./backend/requirements.txt --no-hashes && \
	docker compose build

frontend:
	cd src && chainlit run ./app.py -w

local:
	just bd && docker compose up -d

local-down:
	docker compose down -v