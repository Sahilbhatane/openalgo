# OpenAlgo Branching Strategy Guide

## Overview

This document outlines the **3-branch model** used for OpenAlgo development. Following this strategy ensures code quality, reduces production errors, and enables smooth collaboration across the team.

---

## The 3-Branch Model

```
┌─────────────────────────────────────────────────────────────────┐
│                        BRANCH HIERARCHY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   main (production)     ← Stable, deployed to production        │
│         ▲                                                       │
│         │ PR + Review + Approval                                │
│         │                                                       │
│   staging (test)        ← Integration testing, QA               │
│         ▲                                                       │
│         │ PR + Review                                           │
│         │                                                       │
│   dev/* (features)      ← Active development                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Branch Descriptions

| Branch | Purpose | Deployed To | Protected |
|--------|---------|-------------|-----------|
| `main` | Production-ready code | Production servers | ✅ Yes |
| `staging` | Integration testing & QA | Staging/Test environment | ✅ Yes |
| `dev/*` | Feature development | Local/Dev environment | ❌ No |

---

## Rules for Creating `dev/*` Branches

### Naming Convention

```
dev/<type>/<short-description>
```

#### Branch Types

| Type | Description | Example |
|------|-------------|---------|
| `feature` | New functionality | `dev/feature/groww-broker-integration` |
| `bugfix` | Bug fixes | `dev/bugfix/order-placement-error` |
| `hotfix` | Critical production fixes | `dev/hotfix/api-auth-bypass` |
| `refactor` | Code improvements | `dev/refactor/database-optimization` |
| `docs` | Documentation updates | `dev/docs/api-documentation` |

### Creation Rules

1. **Always branch from `staging`** (not `main`):
   ```bash
   git checkout staging
   git pull origin staging
   git checkout -b dev/feature/your-feature-name
   ```

2. **Keep branch names descriptive but concise** (max 50 characters)

3. **One feature per branch** - avoid combining unrelated changes

4. **Sync regularly with `staging`**:
   ```bash
   git fetch origin
   git rebase origin/staging
   ```

---

## Rules for Merging into `staging`

### Prerequisites

- [ ] All code changes are complete
- [ ] Local tests pass
- [ ] Code follows project style guidelines (run `pre-commit run --all-files`)
- [ ] No merge conflicts with `staging`

### Process

1. **Push your feature branch**:
   ```bash
   git push origin dev/feature/your-feature-name
   ```

2. **Create a Pull Request (PR)** targeting `staging`

3. **PR Requirements**:
   - Clear title describing the change
   - Description explaining what and why
   - Link to related issue (if applicable)
   - Screenshots for UI changes

4. **Code Review**:
   - Minimum **1 approving review** required
   - Address all reviewer comments
   - Re-request review after changes

5. **Merge Strategy**: Use **Squash and Merge** to keep history clean

### Example PR Title Format
```
[Feature] Add Groww broker integration
[Bugfix] Fix order placement timeout issue
[Refactor] Optimize database queries for positions
```

---

## Rules for Merging into `main`

### Prerequisites

- [ ] Changes have been tested on `staging` environment
- [ ] All automated tests pass
- [ ] QA sign-off obtained
- [ ] No critical bugs or regressions

### Process

1. **Create a PR from `staging` to `main`**

2. **PR Requirements**:
   - Summary of all changes included
   - Testing evidence/results
   - Deployment notes (if any)
   - Rollback plan documented

3. **Approvals Required**:
   - Minimum **2 approving reviews**
   - At least one from a senior developer/maintainer

4. **Merge Strategy**: Use **Create a Merge Commit** to preserve history

5. **Post-Merge**:
   - Tag the release: `git tag -a v1.x.x -m "Release v1.x.x"`
   - Update changelog
   - Monitor production for issues

---

## Pull Request (PR) + Review Process

### Creating a Good PR

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Testing Done
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed

## Screenshots (if applicable)
Add screenshots here.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Commented complex logic
- [ ] Updated documentation
```

### Review Guidelines

#### For Reviewers:

1. **Check code quality**:
   - Clean, readable code
   - No hardcoded secrets or credentials
   - Proper error handling

2. **Verify functionality**:
   - Logic is correct
   - Edge cases handled
   - No breaking changes (unless intended)

3. **Security review**:
   - No API keys or secrets in code
   - Input validation present
   - Authentication/authorization correct

4. **Provide constructive feedback**:
   - Be specific about issues
   - Suggest improvements
   - Acknowledge good work

#### Review Response Times:

| Priority | Response Time |
|----------|---------------|
| Critical/Hotfix | Within 2 hours |
| Normal | Within 24 hours |
| Low Priority | Within 48 hours |

---

## Why This Avoids Production Errors

### 1. **Isolation of Changes**
- Development happens in isolated `dev/*` branches
- Untested code never reaches production directly
- Issues are contained and don't affect other developers

### 2. **Multi-Stage Testing**
```
dev/* → staging → main
  │        │        │
  └── Local Tests   │
           └── Integration Tests
                    └── Production Ready
```

### 3. **Code Review as Quality Gate**
- Multiple eyes on every change
- Knowledge sharing across team
- Early bug detection before deployment

### 4. **Staged Rollout**
- Changes are validated in `staging` first
- Real-world testing environment
- Time to catch issues before production impact

### 5. **Easy Rollback**
- Clear history of what was deployed
- Can quickly revert to previous `main` state
- Tagged releases for version tracking

### 6. **Reduced Human Error**
- Protected branches prevent accidental direct commits
- PR requirements ensure process is followed
- Automated checks (CI/CD) catch common issues

---

## Quick Reference Commands

```bash
# Start a new feature
git checkout staging && git pull
git checkout -b dev/feature/my-feature

# Keep feature branch updated
git fetch origin && git rebase origin/staging

# Push feature branch
git push origin dev/feature/my-feature

# After PR is merged, clean up
git checkout staging && git pull
git branch -d dev/feature/my-feature

# Create release tag (after main merge)
git checkout main && git pull
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

---

## Branch Protection Settings (GitHub/GitLab)

### For `main`:
- ✅ Require pull request reviews (2 minimum)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date
- ✅ Include administrators
- ✅ Restrict who can push

### For `staging`:
- ✅ Require pull request reviews (1 minimum)
- ✅ Require status checks to pass
- ✅ Require branches to be up to date

---

## Summary

| Action | Branch | Reviews | Merge Type |
|--------|--------|---------|------------|
| New feature/bugfix | `dev/*` → `staging` | 1 | Squash |
| Release to production | `staging` → `main` | 2 | Merge commit |
| Hotfix (critical) | `dev/hotfix/*` → `main` | 2 | Merge commit |

Following this branching strategy ensures:
- 🛡️ **Stable production** - Only tested code reaches users
- 🔍 **Quality code** - Reviews catch bugs early
- 📝 **Clear history** - Easy to track what changed and when
- 🔄 **Safe rollbacks** - Quick recovery if issues occur
