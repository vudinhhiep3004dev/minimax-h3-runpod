# Worker image (CI-built, public pull)

Built by GitHub Actions — layers pulled on GitHub's network, not from a laptop.

| Registry | Image |
|---|---|
| Public (ttl.sh, 7d) | `ttl.sh/vudinhhiep3004dev-minimax-h3-runpod:7d` |
| GHCR | `ghcr.io/vudinhhiep3004dev/minimax-h3-runpod:latest` |

SHA: `29be43e80d890deeb7902c4c099b1614da4256c6`

Point the Runpod template `imageName` at the public ttl.sh tag (or make GHCR public).
Preferred long-term: console **Import Git Repository** so Runpod builds from this Dockerfile.
