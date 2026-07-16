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
      label: '🚀 Basics',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: '🎯 Fundamentals',
          items: [
            'basics/fundamentals/index',
            'basics/fundamentals/your-first-model',
            'basics/fundamentals/understanding-state',
          ],
        },
        {
          type: 'category',
          label: '🏃 Training Workflows',
          items: [
            'basics/workflows/index',
            'basics/workflows/simple-training',
            'basics/workflows/data-loading-simple',
            'basics/data-streaming',
            'basics/workflows/streaming-data',
            'basics/workflows/observability',
            'basics/workflows/model-export',
          ],
        },
        {
          type: 'category',
          label: '🖼️ Computer Vision',
          items: [
            'basics/vision/index',
            'basics/vision/simple-cnn',
            'basics/vision/resnet-architecture',
          ],
        },
        {
          type: 'category',
          label: '📝 Natural Language Processing',
          items: [
            'basics/text/index',
            'basics/text/simple-transformer',
          ],
        },
        {
          type: 'category',
          label: '📚 Additional Resources',
          collapsed: true,
          items: [
            'basics/checkpointing',
            'basics/training-best-practices',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '🏗️ Architectures',
      items: [
        'architectures/resnet',
        'architectures/bert',
        'architectures/gpt',
      ],
    },
    {
      type: 'category',
      label: '🧩 Applications',
      collapsed: false,
      items: [
        'applications/index',
        {
          type: 'category',
          label: '🎨 Generative Models',
          items: [
            'applications/generative/index',
            'applications/generative/autoencoder',
            'applications/generative/vae',
            'applications/generative/gan',
            'applications/generative/diffusion',
          ],
        },
        {
          type: 'category',
          label: '🔁 Sequence Models & Time Series',
          items: [
            'applications/sequence/index',
            'applications/sequence/recurrent-networks',
          ],
        },
        {
          type: 'category',
          label: '🔬 Graphs, Scientific & Structured',
          items: [
            'applications/scientific/index',
            'applications/scientific/graph-neural-networks',
            'applications/scientific/pinn',
          ],
        },
        {
          type: 'category',
          label: '🧬 Multimodal & Adaptation',
          items: [
            'applications/adaptation/index',
            'applications/adaptation/lora-finetuning',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: '📈 Scale',
      items: [
        'scale/index',
        'scale/data-parallelism',
        'scale/spmd-sharding',
        'scale/pipeline-parallelism',
        'scale/fsdp-fully-sharded',
      ],
    },
    {
      type: 'category',
      label: '🔬 Research',
      items: [
        'research/advanced-techniques',
        'research/custom-training-loops',
        'research/neural-architecture-search',
        'research/adversarial-training',
        'research/curriculum-learning',
        'research/experiment-reproducibility',
        'research/contrastive-learning',
        'research/knowledge-distillation',
        'research/meta-learning',
        'research/reinforcement-learning',
      ],
    },
  ],
};

export default sidebars;
