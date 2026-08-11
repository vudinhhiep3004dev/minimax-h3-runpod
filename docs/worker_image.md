# Worker image (CI-built, public pull)

Built by GitHub Actions — layers pulled on GitHub's network, not from a laptop.

| Registry | Image |
|---|---|
| Public (ttl.sh, 7d) | `ttl.sh/vudinhhiep3004dev-minimax-h3-runpod:7d` |
| GHCR | `ghcr.io/vudinhhiep3004dev/minimax-h3-runpod:latest` |

SHA: `66b0e9a57e6b8501e58d46da5ce3766a89a170a7`

Point the Runpod template `imageName` at the public ttl.sh tag (or make GHCR public).
Preferred long-term: console **Import Git Repository** so Runpod builds from this Dockerfile.
