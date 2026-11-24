FROM python:3.15-slim AS base
WORKDIR /app
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget gcc libc6-dev gpg curl git gawk ripgrep graphviz \
        openjdk-21-jdk-headless maven npm pipx glab pre-commit \
    && rm -rf /var/lib/apt/lists/* && \
    \
    wget -O- https://apt.releases.hashicorp.com/gpg | \
    gpg --dearmor > /usr/share/keyrings/hashicorp-archive-keyring.gpg && \
    \
    gpg --no-default-keyring \
        --keyring /usr/share/keyrings/hashicorp-archive-keyring.gpg \
        --fingerprint && \
    \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
        https://apt.releases.hashicorp.com $(. /etc/os-release && echo $VERSION_CODENAME) main" | \
    tee /etc/apt/sources.list.d/hashicorp.list && \
    \
    apt-get update && \
    apt-get install -y --no-install-recommends \
      terraform \
    && rm -rf /var/lib/apt/lists/*

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
