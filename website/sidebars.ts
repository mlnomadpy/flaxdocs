import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: '🎯 Fundamentals',
      collapsed: false,
      items: [
        'fundamentals/index',
        'fundamentals/your-first-model',
        'fundamentals/understanding-state',
      ],
    },
    {
      type: 'category',
      label: '🏃 Training Workflows',
      collapsed: false,
      items: [
        'workflows/index',
        'workflows/simple-training',
        'workflows/data-loading-simple',
        'workflows/streaming-data',
        'workflows/observability',
        'workflows/model-export',
      ],
    },
    {
      type: 'category',
      label: '🖼️ Computer Vision',
      collapsed: false,
      items: [
        'vision/index',
        'vision/simple-cnn',
        'vision/resnet-architecture',
      ],
    },
    {
      type: 'category',
      label: '📝 Natural Language Processing',
      collapsed: false,
      items: [
        'text/index',
        'text/simple-transformer',
      ],
    },
    {
      type: 'category',
      label: '🚀 Legacy Guides',
      collapsed: true,
      items: [
        'basics/getting-started',
        'basics/model-definition',
        'basics/data-loading',
        'basics/training-loops',
        'basics/checkpointing',
        'basics/training-best-practices',
      ],
    },
    {
      type: 'category',
      label: '📈 Scale',
      items: [
        'scale/distributed-training',
      ],
    },
    {
      type: 'category',
      label: '🔬 Research',
      items: [
        'research/streaming-and-architectures',
        'research/advanced-techniques',
      ],
    },
  ],
};

export default sidebars;
