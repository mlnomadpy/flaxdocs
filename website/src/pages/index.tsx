import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroLeft}>
            <div className={styles.badge}>Production JAX Training</div>
            <Heading as="h1" className={styles.heroTitle}>
              Train Neural Networks
              <br />
              <span className={styles.highlight}>with Flax</span>
            </Heading>
            <p className={styles.heroSubtitle}>
              From basic CNNs to distributed Transformers. Complete recipes for research and production.
            </p>
            <div className={styles.heroButtons}>
              <Link className={styles.primaryButton} to="/docs/basics/fundamentals/your-first-model">
                Start Training
              </Link>
              <Link className={styles.secondaryButton} to="/docs/scale/">
                Scale to Clusters
              </Link>
            </div>
          </div>
          <div className={styles.heroRight}>
            <div className={styles.codePreview}>
              <div className={styles.codeHeader}>
                <span></span>
                <span></span>
                <span></span>
              </div>
              <pre className={styles.codeBlock}>
{`import flax.nnx as nn
import jax.numpy as jnp

class CNN(nn.Module):
  def __init__(self, rngs):
    self.conv = nn.Conv(1, 32, 3, rngs=rngs)
    self.bn = nn.BatchNorm(32, rngs=rngs)
    self.linear = nn.Linear(32, 10, rngs=rngs)
  
  def __call__(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = nn.relu(x)
    return self.linear(x.mean((-2, -1)))`}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function LearningPath() {
  const paths = [
    {
      stage: '01',
      title: 'Basics',
      desc: 'Models, data, training loops',
      items: ['Model Definition', 'Data Loading', 'Training Loops', 'Checkpointing'],
      link: '/docs/basics/fundamentals/your-first-model',
      color: '#2c3e50'
    },
    {
      stage: '02',
      title: 'Scale',
      desc: 'Multi-GPU/TPU training',
      items: ['Data Parallelism', 'Model Parallelism', 'FSDP', 'Pipeline Parallel'],
      link: '/docs/scale/',
      color: '#2563eb'
    },
    {
      stage: '03',
      title: 'Research',
      desc: 'Advanced techniques',
      items: ['Model Export', 'Streaming', 'Observability', 'Custom Loops'],
      link: '/docs/research/advanced-techniques',
      color: '#7c3aed'
    }
  ];

  return (
    <section className={styles.learningPath}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Learning Path</Heading>
          <p>Master Flax training from fundamentals to distributed scale</p>
        </div>
        <div className={styles.pathGrid}>
          {paths.map((path, idx) => (
            <Link key={idx} to={path.link} className={styles.pathCard} style={{'--path-color': path.color} as any}>
              <div className={styles.pathStage}>{path.stage}</div>
              <Heading as="h3" className={styles.pathTitle}>{path.title}</Heading>
              <p className={styles.pathDesc}>{path.desc}</p>
              <ul className={styles.pathItems}>
                {path.items.map((item, i) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
              <span className={styles.pathLink}>Explore →</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function QuickStart() {
  return (
    <section className={styles.quickStart}>
      <div className="container">
        <div className={styles.quickStartContent}>
          <div className={styles.quickStartLeft}>
            <Heading as="h2">Get Started in Minutes</Heading>
            <p>Complete working examples you can run immediately</p>
            <div className={styles.exampleList}>
              <Link to="/docs/basics/fundamentals/your-first-model" className={styles.exampleItem}>
                <span className={styles.exampleIcon}>📦</span>
                <div>
                  <div className={styles.exampleTitle}>Define Models</div>
                  <div className={styles.exampleDesc}>CNNs, RNNs, Transformers</div>
                </div>
              </Link>
              <Link to="/docs/basics/workflows/simple-training" className={styles.exampleItem}>
                <span className={styles.exampleIcon}>⚡</span>
                <div>
                  <div className={styles.exampleTitle}>Training Loops</div>
                  <div className={styles.exampleDesc}>MNIST, ImageNet, LLMs</div>
                </div>
              </Link>
              <Link to="/docs/basics/workflows/data-loading-simple" className={styles.exampleItem}>
                <span className={styles.exampleIcon}>📊</span>
                <div>
                  <div className={styles.exampleTitle}>Data Pipelines</div>
                  <div className={styles.exampleDesc}>TFDS, Grain, Custom</div>
                </div>
              </Link>
            </div>
          </div>
          <div className={styles.quickStartRight}>
            <div className={styles.commandBox}>
              <div className={styles.commandLine}>
                <span className={styles.prompt}>$</span>
                <span>pip install flax jax</span>
              </div>
              <div className={styles.commandLine}>
                <span className={styles.prompt}>$</span>
                <span>python train.py</span>
              </div>
              <div className={styles.output}>
                <div>Epoch 1/10 • Loss: 0.432 • Acc: 87.3%</div>
                <div>Epoch 2/10 • Loss: 0.201 • Acc: 94.1%</div>
                <div>Epoch 3/10 • Loss: 0.145 • Acc: 96.8%</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function Features() {
  const features = [
    {
      title: 'JAX-Native',
      icon: '⚡',
      description: 'Built on JAX for automatic differentiation, JIT compilation, and XLA optimization.',
    },
    {
      title: 'Production Ready',
      icon: '🏭',
      description: 'Battle-tested patterns used at Google and leading ML research labs.',
    },
    {
      title: 'Scale Effortlessly',
      icon: '📈',
      description: 'Single GPU to TPU pods with minimal code changes. Data and model parallelism built-in.',
    },
    {
      title: 'Explicit & Clear',
      icon: '🎯',
      description: 'No hidden state. Full control over RNGs, parameters, and training loops.',
    },
    {
      title: 'Research Focused',
      icon: '🔬',
      description: 'Flexible architecture for experimenting with new ideas and cutting-edge techniques.',
    },
    {
      title: 'Active Community',
      icon: '👥',
      description: 'Join thousands of developers and researchers building with Flax and JAX.',
    },
  ];

  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2">Why Choose Flax NNX?</Heading>
          <p>Modern neural network training with functional programming principles</p>
        </div>
        <div className={styles.featureGrid}>
          {features.map((feature, idx) => (
            <div key={idx} className={styles.featureCard}>
              <div className={styles.featureIcon}>{feature.icon}</div>
              <Heading as="h3" className={styles.featureTitle}>{feature.title}</Heading>
              <p className={styles.featureDesc}>{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Stats() {
  const stats = [
    { number: '20+', label: 'Ready-to-Run Examples' },
    { number: '50+', label: 'Documentation Pages' },
    { number: '5', label: 'Learning Paths' },
    { number: '100%', label: 'Open Source' },
  ];

  return (
    <section className={styles.stats}>
      <div className="container">
        <div className={styles.statsGrid}>
          {stats.map((stat, idx) => (
            <div key={idx} className={styles.statCard}>
              <div className={styles.statNumber}>{stat.number}</div>
              <div className={styles.statLabel}>{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CallToAction() {
  return (
    <section className={styles.cta}>
      <div className="container">
        <div className={styles.ctaContent}>
          <Heading as="h2" className={styles.ctaTitle}>
            Ready to Start Training?
          </Heading>
          <p className={styles.ctaSubtitle}>
            Join the community of developers and researchers using Flax for neural network training
          </p>
          <div className={styles.ctaButtons}>
            <Link className={styles.primaryButton} to="/docs/basics/fundamentals/your-first-model">
              Get Started Now
            </Link>
            <Link className={styles.secondaryButton} to="https://github.com/mlnomadpy/flaxdocs/tree/main/examples">
              Browse Examples
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Production-grade JAX training recipes from basics to distributed scale">
      <HomepageHeader />
      <Features />
      <LearningPath />
      <QuickStart />
      <Stats />
      <CallToAction />
    </Layout>
  );
}
