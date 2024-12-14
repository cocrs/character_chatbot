alias l := local
alias ld := local-down

alias bd := build-docker
alias f := frontend

build-docker:
	cd backend && \
	if ! pdm lock --check; then \
		echo "Dependencies have changed, updating lock file..."; \
		pdm lock --update-reuse; \
		pdm export -o requirements.txt --no-hashes; \
	else \
		echo "Lock file is up-to-date. Skipping update."; \
	fi; \
	docker compose build

frontend:
	cd src && chainlit run ./app.py -w

local:
	just bd && docker compose up -d

local-down:
	docker compose down -v