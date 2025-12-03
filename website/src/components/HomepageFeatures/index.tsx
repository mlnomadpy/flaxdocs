import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
  link: string;
  linkText: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Build Models',
    icon: '⚡',
    description: (
      <>
        Define CNNs, Transformers, and custom architectures with Flax NNX.
        Learn state management, initialization patterns, and checkpoint strategies.
      </>
    ),
    link: '/docs/basics/model-definition',
    linkText: 'Model Definition →',
  },
  {
    title: 'Load Data',
    icon: '📊',
    description: (
      <>
        Stream datasets with Grain, integrate TFDS, or build custom dataloaders.
        Master batching, prefetching, and multi-host data parallelism.
      </>
    ),
    link: '/docs/basics/data-loading',
    linkText: 'Data Loading →',
  },
  {
    title: 'Train at Scale',
    icon: '🚀',
    description: (
      <>
        Train on 100+ GPUs/TPUs with JAX sharding APIs. Implement FSDP,
        tensor parallelism, and pipeline parallelism for large models.
      </>
    ),
    link: '/docs/scale/distributed-training',
    linkText: 'Distributed Training →',
  },
];

const QuickLinks = [
  { label: 'Checkpointing', link: '/docs/basics/checkpointing' },
  { label: 'Training Loops', link: '/docs/basics/training-loops' },
  { label: 'Model Export', link: '/docs/research/model-export' },
  { label: 'Observability', link: '/docs/research/observability' },
];

function Feature({title, icon, description, link, linkText}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <Link to={link} className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
        <span className={styles.featureLink}>{linkText}</span>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <>
      <section className={styles.featuresSection}>
        <div className="container">
          <div className="row">
            {FeatureList.map((props, idx) => (
              <Feature key={idx} {...props} />
            ))}
          </div>
        </div>
      </section>
      
      <section className={styles.quickLinksSection}>
        <div className="container">
          <Heading as="h2" className={styles.sectionTitle}>Quick Access</Heading>
          <div className={styles.quickLinksGrid}>
            {QuickLinks.map((item, idx) => (
              <Link key={idx} to={item.link} className={styles.quickLinkCard}>
                {item.label}
                <span className={styles.arrow}>→</span>
              </Link>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
