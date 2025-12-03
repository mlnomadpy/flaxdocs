# Deployment Guide

This document explains how the Flax Training Documentation is deployed to GitHub Pages.

## Overview

The documentation is built with Docusaurus and automatically deployed to GitHub Pages using GitHub Actions whenever changes are pushed to the `main` branch.

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Push to    │         │ GitHub       │                 │
│  │   main       │────────>│ Actions      │                 │
│  │   branch     │         │ Workflow     │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                   │                          │
│                          ┌────────▼────────┐                │
│                          │  Build Docs     │                │
│                          │  npm run build  │                │
│                          └────────┬────────┘                │
│                                   │                          │
│                          ┌────────▼────────┐                │
│                          │  Deploy to      │                │
│                          │  GitHub Pages   │                │
│                          └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    https://mlnomadpy.github.io/flaxdocs/
```

## GitHub Actions Workflow

The deployment is handled by `.github/workflows/deploy.yml`:

### Build Job
1. **Checkout**: Clones the repository
2. **Setup Node.js**: Installs Node.js 20
3. **Install Dependencies**: Runs `npm ci` in the `website/` directory
4. **Build Website**: Runs `npm run build` to generate static files
5. **Upload Artifacts**: Uploads the build output for deployment

### Deploy Job
1. **Deploy to GitHub Pages**: Uses the official `deploy-pages` action
2. **Only runs on**: Push to `main` branch (not on PRs)
3. **Environment**: Uses `github-pages` environment with proper permissions

## Manual Deployment

If you need to deploy manually:

### Prerequisites
- Repository settings → Pages → Source set to "GitHub Actions"
- Workflow permissions set to "Read and write permissions"

### Steps

1. **Build locally**:
   ```bash
   cd website
   npm install
   npm run build
   ```

2. **Verify build**:
   ```bash
   npm run serve
   # Opens at http://localhost:3000
   ```

3. **Commit and push to main**:
   ```bash
   git add .
   git commit -m "Update documentation"
   git push origin main
   ```

4. **Monitor deployment**:
   - Go to Actions tab in GitHub
   - Watch the "Deploy to GitHub Pages" workflow
   - Once complete, site is live at: https://mlnomadpy.github.io/flaxdocs/

## Configuration

### Docusaurus Config

Key settings in `website/docusaurus.config.ts`:

```typescript
{
  url: 'https://mlnomadpy.github.io',
  baseUrl: '/flaxdocs/',
  organizationName: 'mlnomadpy',
  projectName: 'flaxdocs',
  trailingSlash: false,
  onBrokenLinks: 'throw',
}
```

### GitHub Pages Settings

Required repository settings:
- **Settings → Pages → Source**: GitHub Actions
- **Settings → Actions → General → Workflow permissions**: Read and write permissions
- **Settings → Environments**: `github-pages` environment should be auto-created

## Troubleshooting

### Build Fails

**Check for broken links**:
```bash
cd website
npm run build
```

If there are broken links, the build will fail with details about which pages have broken links.

**Fix broken links**:
- Update internal links to use correct paths
- Remove links to non-existent pages
- Use relative paths for internal documentation

### Deployment Fails

**Permission Issues**:
- Check Settings → Actions → General → Workflow permissions
- Ensure "Read and write permissions" is selected

**Pages Not Enabled**:
- Go to Settings → Pages
- Ensure Source is set to "GitHub Actions"

**Wrong Base URL**:
- Verify `baseUrl` in `docusaurus.config.ts` matches repository name
- Should be `/flaxdocs/` for this project

### 404 on Deployed Site

**Check baseUrl**:
- Must match the repository name
- Should be `/flaxdocs/` not `/`

**Check trailing slash**:
- Set `trailingSlash: false` in config
- Or ensure all links use consistent trailing slash behavior

### CSS Not Loading

**Check asset paths**:
- All assets should use relative paths
- Docusaurus handles this automatically if baseUrl is correct

## Monitoring

### Check Deployment Status

1. **GitHub Actions**:
   - Go to Actions tab
   - View workflow runs
   - Check logs for any errors

2. **GitHub Pages**:
   - Settings → Pages shows deployment status
   - Shows when last deployment occurred
   - Provides site URL

3. **Live Site**:
   - Visit https://mlnomadpy.github.io/flaxdocs/
   - Verify pages load correctly
   - Check navigation works

## Development Workflow

### Local Development

```bash
cd website
npm start
# Opens at http://localhost:3000/flaxdocs/
```

### Adding New Documentation

1. Create markdown file in `website/docs/`
2. Update `website/sidebars.ts` if needed
3. Test locally with `npm start`
4. Build to verify: `npm run build`
5. Commit and push to trigger deployment

### Preview Changes

**On Pull Requests**:
- Workflow runs but doesn't deploy
- Check build succeeds
- Review changes locally

**On Main Branch**:
- Workflow runs and deploys
- Changes go live automatically
- Usually takes 2-3 minutes

## Performance

### Build Time
- Initial build: ~60 seconds
- Incremental rebuild: ~2 seconds (local dev)

### Deployment Time
- Total workflow: ~2-3 minutes
- Build job: ~1-2 minutes
- Deploy job: ~30-60 seconds

### Optimization Tips

1. **Cache Dependencies**:
   - Workflow already caches npm packages
   - Speeds up subsequent builds

2. **Minimize Assets**:
   - Compress images before adding
   - Use WebP format when possible

3. **Code Splitting**:
   - Docusaurus handles this automatically
   - Each page loads only needed JavaScript

## Backup and Recovery

### Backup Documentation

```bash
# Clone repository
git clone https://github.com/mlnomadpy/flaxdocs.git

# All documentation is in website/docs/
cd flaxdocs/website/docs
```

### Restore from Backup

```bash
# Push to main branch to trigger rebuild
git push origin main
```

### Rollback Deployment

```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Or reset to specific commit
git reset --hard <commit-hash>
git push origin main --force
```

## Security

### Automated Checks

- CodeQL scans for vulnerabilities
- Dependabot alerts for outdated packages
- No secrets in repository

### Update Dependencies

```bash
cd website
npm audit
npm audit fix
# Or update specific packages
npm update
```

## Support

For issues with deployment:

1. Check workflow logs in Actions tab
2. Review this deployment guide
3. Open an issue in the repository
4. Contact repository maintainers

---

Last updated: December 2, 2025
