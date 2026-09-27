FROM python:3.14-slim AS base
WORKDIR /app
RUN apt-get update \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*

ARG USER
USER $USER

RUN which obsidian
RUN obsidian --no-sandbox --user-data-dir=/tmp --disable-setuid-sandbox vaults

FROM base AS development
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix]'

FROM base AS bot
COPY . .
RUN pip install --no-cache-dir '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix]'
CMD ["python", "-m", "karl", "bot"]

FROM base AS mlx
COPY . .
RUN pip install --no-cache-dir '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix]'
CMD ["python", "-m", "karl.mlx.server"]
