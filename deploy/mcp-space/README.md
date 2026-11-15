---
title: Alpha Trend MCP
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Alpha Trend MCP (remote)

Serves the dashboard's read-only MCP tools at `https://<this-space>.hf.space/<MCP_TOKEN>/mcp`.

Space **secrets** (Settings → Variables and secrets):

| Name | Needed | What |
|---|---|---|
| `MCP_TOKEN` | yes | 24+ random letters and digits. It is the password in the URL. |
| `GITHUB_TOKEN` | for your book | Read-only token for the dashboard repo, to read holdings and watchlist. |
| `GITHUB_REPO` | for your book | `owner/nifty-dashboard` |
| `REPO_URL` | private repo only | `https://<token>@github.com/owner/repo.git` |

The files in this folder are the whole Space: `Dockerfile` and this `README.md`. The container
clones the dashboard repo at start and pulls it every 30 minutes.
