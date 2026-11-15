# Alpha Trend MCP server as a Hugging Face Docker Space (or any container host).
# The image holds only Python packages; the repo is cloned when the container starts, so a
# restart always serves the latest data.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "mcp>=2.0" pandas pyarrow numpy requests uvicorn openpyxl
RUN useradd -m -u 1000 app
USER app
WORKDIR /home/app
ENV PORT=7860 REFRESH_MINUTES=30 PYTHONUNBUFFERED=1
# REPO_URL: the dashboard repo. For a private repo use https://<token>@github.com/owner/repo.git (as a secret).
ENV REPO_URL=https://github.com/adi-git01/nifty-dashboard.git
EXPOSE 7860
CMD ["sh", "-c", "rm -rf repo && git clone --depth 1 \"$REPO_URL\" repo && cd repo && exec python -m mcp_server.remote"]
