# Git Instructions for Gitea Instance



## Clone Repository

```bash
git clone http://claude%40bitwarelabs.com:henke12345@172.21.0.1:3001/claude/REPO_NAME.git
```

## Add Remote to Existing Repository

```bash
git remote add origin http://claude%40bitwarelabs.com:henke12345@172.21.0.1:3001/claude/REPO_NAME.git
```

## Push to Repository

```bash
git push -u origin master
# or for main branch
git push -u origin main
```

## Pull from Repository

```bash
git pull origin master
```

## Create New Repository via API

```bash
curl -X POST http://172.21.0.1:3001/api/v1/user/repos \
  -u "claude@bitwarelabs.com:henke12345" \
  -H "Content-Type: application/json" \
  -d '{"name": "repo-name", "description": "Repository description", "private": false}'
```

## Notes

- The `@` symbol in the email is URL-encoded as `%40` in git URLs
- Use HTTP authentication with username:password in the URL
- Repository names should be lowercase with hyphens
- Web interface available at http://172.21.0.1:3001

## Existing Repositories

- **bitwarelabs-website**: http://172.21.0.1:3001/claude/bitwarelabs-website
- **test-repo**: http://172.21.0.1:3001/claude/test-repo
- **test-website**: http://172.21.0.1:3001/claude/test-website
