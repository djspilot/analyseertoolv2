# GitHub Setup

I can’t push to GitHub directly, but the project is ready to be published. Follow the steps below.

## Option A — GitHub CLI (fast)

```bash
gh repo create analyseertoolv2 --private --source . --remote origin --push
```

## Option B — Manual

1. Create a new repository on GitHub.
2. In this repo, run:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/djspilot/analyseertoolv2.git
git push -u origin master
```

✅ **Done!** Repository published at: https://github.com/djspilot/analyseertoolv2
