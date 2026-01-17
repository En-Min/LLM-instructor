import { motion } from 'framer-motion';
import './WelcomeScreen.css';

interface WelcomeScreenProps {
  onStartLearning: (functionName: string) => void;
  totalFunctions: number;
  completedFunctions: number;
}

const suggestedStarts = [
  {
    name: 'linear',
    displayName: 'run_linear',
    description: 'Start with the basics - matrix multiplication',
    difficulty: 'Easy',
    emoji: '🎯',
  },
  {
    name: 'embedding',
    displayName: 'run_embedding',
    description: 'Learn how tokens become vectors',
    difficulty: 'Easy',
    emoji: '📚',
  },
  {
    name: 'scaled_dot_product_attention',
    displayName: 'run_attention',
    description: 'The heart of transformers',
    difficulty: 'Medium',
    emoji: '⚡',
  },
];

const learningPath = [
  { phase: 'Foundation', icon: '🧱', count: 6 },
  { phase: 'Feed-Forward', icon: '🔀', count: 1 },
  { phase: 'Attention', icon: '👁️', count: 4 },
  { phase: 'Transformer', icon: '🏗️', count: 2 },
  { phase: 'Training', icon: '🎓', count: 4 },
  { phase: 'Tokenization', icon: '✂️', count: 4 },
];

export function WelcomeScreen({ onStartLearning, totalFunctions, completedFunctions }: WelcomeScreenProps) {
  return (
    <div className="welcome-screen">
      {/* Animated background */}
      <div className="welcome-bg">
        <div className="bg-gradient-1" />
        <div className="bg-gradient-2" />
        <div className="bg-grid" />
      </div>

      <div className="welcome-content">
        {/* Hero Section */}
        <motion.div
          className="welcome-hero"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        >
          <div className="hero-badge">
            <span className="badge-icon">🧠</span>
            <span className="badge-text">Interactive Learning</span>
          </div>

          <h1 className="hero-title">
            Build a Transformer
            <span className="title-accent"> From Scratch</span>
          </h1>

          <p className="hero-subtitle">
            Master the fundamentals of Large Language Models through hands-on implementation.
            Your Socratic AI tutor will guide you step by step.
          </p>

          <div className="hero-stats">
            <div className="stat">
              <span className="stat-value">{totalFunctions}</span>
              <span className="stat-label">Functions to Build</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-value">6</span>
              <span className="stat-label">Learning Phases</span>
            </div>
            <div className="stat-divider" />
            <div className="stat">
              <span className="stat-value">{completedFunctions}</span>
              <span className="stat-label">Completed</span>
            </div>
          </div>
        </motion.div>

        {/* Learning Path Visualization */}
        <motion.div
          className="learning-path-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
        >
          <h2 className="section-title">Your Learning Journey</h2>
          <div className="learning-path">
            {learningPath.map((phase, index) => (
              <motion.div
                key={phase.phase}
                className="path-node"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.4 + index * 0.1 }}
              >
                <div className="node-icon">{phase.icon}</div>
                <div className="node-info">
                  <span className="node-name">{phase.phase}</span>
                  <span className="node-count">{phase.count} functions</span>
                </div>
                {index < learningPath.length - 1 && (
                  <div className="path-connector">
                    <svg width="40" height="12" viewBox="0 0 40 12">
                      <path
                        d="M0 6 L30 6 M25 1 L30 6 L25 11"
                        stroke="currentColor"
                        strokeWidth="2"
                        fill="none"
                      />
                    </svg>
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Quick Start Cards */}
        <motion.div
          className="quick-start-section"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4, ease: 'easeOut' }}
        >
          <h2 className="section-title">Start Learning</h2>
          <p className="section-subtitle">Choose a function to begin your journey</p>

          <div className="start-cards">
            {suggestedStarts.map((item, index) => (
              <motion.button
                key={item.name}
                className="start-card"
                onClick={() => onStartLearning(item.name)}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.6 + index * 0.1 }}
                whileHover={{ scale: 1.02, y: -4 }}
                whileTap={{ scale: 0.98 }}
              >
                <span className="card-emoji">{item.emoji}</span>
                <div className="card-content">
                  <h3 className="card-title">{item.displayName}</h3>
                  <p className="card-description">{item.description}</p>
                </div>
                <span className={`card-difficulty ${item.difficulty.toLowerCase()}`}>
                  {item.difficulty}
                </span>
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Features */}
        <motion.div
          className="features-section"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.8 }}
        >
          <div className="feature">
            <span className="feature-icon">💬</span>
            <span className="feature-text">Socratic Dialogue</span>
          </div>
          <div className="feature">
            <span className="feature-icon">🧪</span>
            <span className="feature-text">Auto Test Running</span>
          </div>
          <div className="feature">
            <span className="feature-icon">💡</span>
            <span className="feature-text">Progressive Hints</span>
          </div>
          <div className="feature">
            <span className="feature-icon">📊</span>
            <span className="feature-text">Progress Tracking</span>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
