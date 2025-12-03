# Project Structure

This document outlines the structure of the Flax Training Documentation website.

## Directory Layout

```
flaxdocs/
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions CI/CD workflow
├── website/                      # Docusaurus website root
│   ├── blog/                    # Blog posts (optional)
│   ├── docs/                    # Main documentation
│   │   ├── intro.md            # Landing page
│   │   ├── basics/             # Basic training guides
│   │   │   ├── getting-started.md
│   │   │   ├── training-best-practices.md
│   │   │   └── checkpointing.md
│   │   ├── scale/              # Scaling guides
│   │   │   └── distributed-training.md
│   │   └── research/           # Advanced research techniques
│   │       └── advanced-techniques.md
│   ├── src/                    # Custom React components
│   │   ├── components/
│   │   │   └── HomepageFeatures/
│   │   ├── css/
│   │   │   └── custom.css
│   │   └── pages/
│   │       ├── index.tsx       # Home page
│   │       ├── index.module.css
│   │       └── markdown-page.md
│   ├── static/                 # Static assets
│   │   └── img/               # Images and icons
│   ├── docusaurus.config.ts   # Main configuration
│   ├── sidebars.ts            # Sidebar navigation
│   ├── package.json           # Node dependencies
│   └── tsconfig.json          # TypeScript config
├── DEPLOYMENT.md              # Deployment guide
├── README.md                  # Project README
└── LICENSE                    # MIT License
```

## Key Files

### Configuration

- **docusaurus.config.ts**: Main Docusaurus configuration
  - Site metadata (title, tagline, favicon)
  - GitHub Pages settings (url, baseUrl, organizationName)
  - Theme configuration (navbar, footer, color mode)
  - Plugin settings (docs, blog)
  - Broken link validation

- **sidebars.ts**: Documentation sidebar structure
  - Defines navigation hierarchy
  - Groups documentation by section
  - Controls page ordering

- **package.json**: Node.js dependencies and scripts
  - Docusaurus core and preset
  - React and React DOM
  - TypeScript types
  - Build and dev scripts

### Documentation

- **docs/intro.md**: Main landing page
  - Overview of documentation structure
  - Links to all sections
  - Getting started information

- **docs/basics/**: Fundamental training guides
  - **getting-started.md**: Installation and first model (2,137 lines)
  - **training-best-practices.md**: Optimization techniques (297 lines)
  - **checkpointing.md**: Saving and restoring models (351 lines)

- **docs/scale/**: Scaling and distributed training
  - **distributed-training.md**: Multi-device training (420 lines)
  - Covers data parallelism, model parallelism, pmap, pjit
  - Performance optimization techniques

- **docs/research/**: Advanced research techniques
  - **advanced-techniques.md**: Cutting-edge methods (507 lines)
  - Contrastive learning, meta-learning, NAS
  - Adversarial training, knowledge distillation
  - Curriculum learning and reproducibility

### React Components

- **src/pages/index.tsx**: Home page component
  - Hero section with title and tagline
  - Call-to-action button
  - Features section

- **src/components/HomepageFeatures/index.tsx**: Feature cards
  - Three main sections: Basics, Scale, Research
  - Custom descriptions for Flax training
  - SVG icons for visual appeal

### CI/CD

- **.github/workflows/deploy.yml**: Deployment workflow
  - Triggers on push to main and PRs
  - Build job: Install deps, build site, upload artifacts
  - Deploy job: Deploy to GitHub Pages (main only)
  - Proper permissions and concurrency control

## Documentation Sections

### 🚀 Basics (3 guides)

**Purpose**: Get started with Flax training  
**Target Audience**: Beginners to intermediate users  
**Topics**:
- Environment setup and installation
- Creating first model with Flax
- Training loops and optimization
- Best practices for training
- Model checkpointing and evaluation

### 📈 Scale (1 guide, expandable)

**Purpose**: Scale models to production  
**Target Audience**: Intermediate to advanced users  
**Topics**:
- Distributed training across devices
- Data parallelism with pmap
- Model parallelism with pjit
- Multi-host training setups
- Performance optimization

### 🔬 Research (1 guide, expandable)

**Purpose**: Advanced research techniques  
**Target Audience**: Researchers and advanced practitioners  
**Topics**:
- Custom training loops and states
- Contrastive learning (SimCLR)
- Meta-learning (MAML)
- Neural Architecture Search (DARTS)
- Adversarial training
- Knowledge distillation
- Curriculum learning

## Adding New Documentation

### Create New Guide

1. **Choose section**: basics, scale, or research
2. **Create markdown file**: `docs/{section}/{name}.md`
3. **Add front matter**:
   ```markdown
   ---
   sidebar_position: 2
   ---
   
   # Your Title
   ```
4. **Write content** with code examples
5. **Update sidebars.ts** if needed
6. **Test locally**: `npm start` in website/
7. **Build**: `npm run build` to verify

### Markdown Features

- **Code blocks** with syntax highlighting
- **Admonitions** (notes, warnings, tips)
- **MDX components** for interactive content
- **Images** from static folder
- **Internal links** with automatic validation
- **Front matter** for metadata

### Best Practices

1. Use clear, descriptive titles
2. Include practical code examples
3. Add links to related documentation
4. Test all code snippets
5. Use proper markdown syntax
6. Follow existing style guide
7. Build locally before committing

## Static Assets

### Images

Location: `website/static/img/`

- **favicon.ico**: Site favicon
- **logo.svg**: Site logo
- **docusaurus-*.svg**: Feature section icons
- **docusaurus-social-card.jpg**: Social media preview

### Adding Images

1. Place in `static/img/`
2. Reference in markdown: `![Alt text](/img/image.png)`
3. Use relative paths for docs: `![Alt](/img/image.png)`

## Build Output

When built (`npm run build`), generates:

```
website/build/
├── index.html          # Home page
├── docs.html           # Docs landing
├── docs/               # Documentation pages
│   ├── basics/
│   ├── scale/
│   └── research/
├── blog/               # Blog pages
├── assets/             # Compiled CSS/JS
├── img/                # Static images
└── sitemap.xml         # SEO sitemap
```

## Maintenance

### Regular Tasks

- **Update dependencies**: `npm update` in website/
- **Check for broken links**: `npm run build`
- **Review and merge PRs**: Ensure quality
- **Update documentation**: Keep current with Flax releases
- **Monitor deployment**: Check GitHub Actions

### Dependencies

Key packages:
- `@docusaurus/core`: 3.9.2
- `@docusaurus/preset-classic`: 3.9.2
- `react`: ^19.0.0
- `typescript`: ~5.6.2

Update with:
```bash
cd website
npm outdated        # Check for updates
npm update          # Update minor versions
npm audit fix       # Fix vulnerabilities
```

## Deployment

### Automatic

- Push to `main` branch → Deploys automatically
- View progress in Actions tab
- Live at: https://mlnomadpy.github.io/flaxdocs/

### Manual

```bash
cd website
npm run build
# Upload build/ folder to any static host
```

## Performance

### Metrics

- **Build time**: ~60 seconds
- **Page load**: <1 second (after first load)
- **Lighthouse score**: 90+ (all categories)

### Optimization

- Code splitting by route
- Image optimization
- CSS/JS minification
- Lazy loading
- Service worker for offline support

---

For more details, see [DEPLOYMENT.md](DEPLOYMENT.md) and [README.md](README.md).
