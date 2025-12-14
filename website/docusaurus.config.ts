import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';


// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'Flax Training Docs',
  tagline: 'Learn how to train neural networks with Flax',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://mlnomadpy.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/flaxdocs/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'mlnomadpy', // Usually your GitHub org/user name.
  projectName: 'flaxdocs', // Usually your repo name.
  trailingSlash: false,

  onBrokenLinks: 'throw',

  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  headTags: [
    // Structured data for better search engine understanding
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org/',
        '@type': 'WebSite',
        name: 'Flax Training Docs',
        description: 'Comprehensive guide to training neural networks with Flax NNX and JAX',
        url: 'https://mlnomadpy.github.io/flaxdocs/',
        potentialAction: {
          '@type': 'SearchAction',
          target: 'https://mlnomadpy.github.io/flaxdocs/?q={search_term_string}',
          'query-input': 'required name=search_term_string',
        },
      }),
    },
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org/',
        '@type': 'Organization',
        name: 'Flax Training Docs',
        url: 'https://mlnomadpy.github.io/flaxdocs/',
        logo: 'https://mlnomadpy.github.io/flaxdocs/img/logo.svg',
        description: 'Educational resource for learning Flax NNX and JAX neural network training',
        sameAs: [
          'https://github.com/mlnomadpy/flaxdocs',
        ],
      }),
    },
  ],

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          editUrl:
            'https://github.com/mlnomadpy/flaxdocs/tree/main/website/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          breadcrumbs: true,
          docItemComponent: '@theme/DocItem',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/mlnomadpy/flaxdocs/tree/main/website/',
          // Useful options to enforce blogging best practices
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    // Global metadata for SEO
    metadata: [
      {name: 'keywords', content: 'Flax, JAX, neural networks, machine learning, deep learning, NNX, training, distributed training, TPU, GPU, PyTorch alternative, TensorFlow alternative'},
      {name: 'description', content: 'Comprehensive guide to training neural networks with Flax NNX and JAX. Learn distributed training, model optimization, and production-ready ML workflows.'},
      {name: 'twitter:card', content: 'summary_large_image'},
      {name: 'twitter:title', content: 'Flax Training Docs - Master Neural Network Training with JAX'},
      {name: 'twitter:description', content: 'Learn Flax NNX and JAX for neural network training. From basics to distributed training at scale.'},
      {name: 'og:type', content: 'website'},
      {name: 'og:title', content: 'Flax Training Docs - Neural Network Training with JAX'},
      {name: 'og:description', content: 'Comprehensive guide to training neural networks with Flax NNX and JAX. Learn distributed training, model optimization, and production-ready ML workflows.'},
      {name: 'og:locale', content: 'en_US'},
    ],
    // Announcement bar for important updates
    announcementBar: {
      id: 'new_examples',
      content:
        '⭐️ Check out our <a target="_blank" href="https://github.com/mlnomadpy/flaxdocs/tree/main/examples">20+ runnable examples</a> for Flax NNX training!',
      backgroundColor: '#2563eb',
      textColor: '#ffffff',
      isCloseable: true,
    },
    colorMode: {
      respectPrefersColorScheme: true,
      defaultMode: 'light',
      disableSwitch: false,
    },
    // Table of contents settings
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    navbar: {
      title: 'Flax Training Docs',
      logo: {
        alt: 'Flax Logo',
        src: 'img/logo.svg',
        width: 32,
        height: 32,
      },
      hideOnScroll: false,
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: '📚 Documentation',
        },
        {
          type: 'dropdown',
          label: '🚀 Quick Links',
          position: 'left',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/basics/fundamentals/your-first-model',
            },
            {
              label: 'Examples Repository',
              href: 'https://github.com/mlnomadpy/flaxdocs/tree/main/examples',
            },
            {
              label: 'Training Best Practices',
              to: '/docs/basics/training-best-practices',
            },
            {
              label: 'Distributed Training',
              to: '/docs/scale/',
            },
          ],
        },
        {to: '/blog', label: '📝 Blog', position: 'left'},
        {
          type: 'search',
          position: 'right',
        },
        {
          href: 'https://github.com/mlnomadpy/flaxdocs',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Getting Started',
              to: '/docs',
            },
            {
              label: 'Fundamentals',
              to: '/docs/basics/fundamentals',
            },
            {
              label: 'Training Workflows',
              to: '/docs/basics/workflows',
            },
            {
              label: 'Scale to Production',
              to: '/docs/scale/',
            },
            {
              label: 'Research Techniques',
              to: '/docs/research/streaming-and-architectures',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/mlnomadpy/flaxdocs/discussions',
            },
            {
              label: 'Flax GitHub',
              href: 'https://github.com/google/flax',
            },
            {
              label: 'Flax Official Docs',
              href: 'https://flax.readthedocs.io/',
            },
            {
              label: 'JAX Documentation',
              href: 'https://jax.readthedocs.io/',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Examples Repository',
              href: 'https://github.com/mlnomadpy/flaxdocs/tree/main/examples',
            },
            {
              label: 'GitHub Issues',
              href: 'https://github.com/mlnomadpy/flaxdocs/issues',
            },
          ],
        },
        {
          title: 'Legal',
          items: [
            {
              label: 'MIT License',
              href: 'https://github.com/mlnomadpy/flaxdocs/blob/main/LICENSE',
            },
            {
              label: 'Contribute',
              href: 'https://github.com/mlnomadpy/flaxdocs/blob/main/README.md#-contributing',
            },
          ],
        },
      ],
      logo: {
        alt: 'Flax Training Docs Logo',
        src: 'img/logo.svg',
        width: 160,
        height: 51,
      },
      copyright: `Copyright © ${new Date().getFullYear()} Flax Training Docs. Built with ❤️ using Docusaurus.`,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'yaml', 'toml', 'diff'],
      defaultLanguage: 'python',
      magicComments: [
        {
          className: 'theme-code-block-highlighted-line',
          line: 'highlight-next-line',
          block: {start: 'highlight-start', end: 'highlight-end'},
        },
        {
          className: 'code-block-error-line',
          line: 'error-next-line',
        },
      ],
    },
  } satisfies Preset.ThemeConfig,
    stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
};

export default config;
