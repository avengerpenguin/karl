FROM python:3.14-slim AS base
WORKDIR /app
RUN apt-get update \
    && apt-get install -y git maven openjdk-21-jdk gawk glab pre-commit pipx npm graphviz \
    && rm -rf /var/lib/apt/lists/*

RUN pip install analytics-mcp

FROM base AS development
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix,mysql,google]'
ARG USER
USER $USER

FROM base AS bot
COPY . .
RUN pip install --no-cache-dir '.[imap,beeper,todoist,gitlab,jira,confluence,tavily,matrix,mysql,google]'
CMD ["python", "-m", "karl", "bot"]
ARG USER
USER $USER
