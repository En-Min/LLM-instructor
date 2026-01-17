import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { AssignmentPhase, FunctionStatus } from '../types';
import './ProgressSidebar.css';

interface ProgressSidebarProps {
  phases: AssignmentPhase[];
  currentFunction?: string;
  onFunctionSelect?: (functionName: string) => void;
}

const statusIcons: Record<FunctionStatus, string> = {
  not_started: '○',
  in_progress: '◐',
  passed: '●',
  failed: '✕',
};

const statusColors: Record<FunctionStatus, string> = {
  not_started: 'var(--color-text-tertiary)',
  in_progress: 'var(--color-pending)',
  passed: 'var(--color-success)',
  failed: 'var(--color-error)',
};

export function ProgressSidebar({ phases, currentFunction, onFunctionSelect }: ProgressSidebarProps) {
  const [expandedPhases, setExpandedPhases] = useState<Set<string>>(
    new Set(phases.map(p => p.name)) // Start with all phases expanded
  );

  const totalFunctions = phases.reduce((sum, phase) => sum + phase.functions.length, 0);
  const completedFunctions = phases.reduce(
    (sum, phase) => sum + phase.functions.filter(f => f.status === 'passed').length,
    0
  );
  const progressPercentage = (completedFunctions / totalFunctions) * 100;

  const togglePhase = (phaseName: string) => {
    setExpandedPhases(prev => {
      const next = new Set(prev);
      if (next.has(phaseName)) {
        next.delete(phaseName);
      } else {
        next.add(phaseName);
      }
      return next;
    });
  };

  return (
    <aside className="progress-sidebar">
      {/* Header */}
      <motion.div
        className="sidebar-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.1 }}
      >
        <h2 className="sidebar-title">Assignment 1</h2>
        <p className="sidebar-subtitle">CS336: Language Modeling</p>

        {/* Progress bar */}
        <div className="progress-wrapper">
          <div className="progress-stats">
            <span className="progress-label">Progress</span>
            <span className="progress-value">{completedFunctions}/{totalFunctions}</span>
          </div>
          <div className="progress-bar">
            <motion.div
              className="progress-fill"
              initial={{ width: 0 }}
              animate={{ width: `${progressPercentage}%` }}
              transition={{ duration: 0.8, delay: 0.3 }}
            />
          </div>
        </div>
      </motion.div>

      {/* Function list */}
      <div className="function-list">
        {phases.map((phase, phaseIndex) => {
          const isExpanded = expandedPhases.has(phase.name);
          const phaseCompleted = phase.functions.filter(f => f.status === 'passed').length;
          const phaseTotal = phase.functions.length;

          return (
            <motion.div
              key={phase.name}
              className="phase-section"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.2 + phaseIndex * 0.1 }}
            >
              <div className="phase-header" onClick={() => togglePhase(phase.name)}>
                <h3 className="phase-title">{phase.name}</h3>
                <div className="phase-stats">
                  <span className="phase-count">
                    {phaseCompleted}/{phaseTotal}
                  </span>
                  <motion.span
                    className="phase-chevron"
                    animate={{ rotate: isExpanded ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    ▼
                  </motion.span>
                </div>
              </div>

              <AnimatePresence>
                {isExpanded && (
                  <motion.ul
                    className="function-items"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    {phase.functions.map((func, funcIndex) => {
                      const isActive = currentFunction === func.name;
                      const delay = 0.3 + phaseIndex * 0.1 + funcIndex * 0.05;

                      return (
                        <motion.li
                          key={func.name}
                          className={`function-item ${isActive ? 'active' : ''}`}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.4, delay }}
                          onClick={() => onFunctionSelect?.(func.name)}
                        >
                          <span
                            className="function-status"
                            style={{ color: statusColors[func.status] }}
                          >
                            {statusIcons[func.status]}
                          </span>
                          <div className="function-info">
                            <span className="function-name">{func.displayName}</span>
                            {func.attempts > 0 && (
                              <span className="function-meta">
                                {func.attempts} {func.attempts === 1 ? 'attempt' : 'attempts'}
                              </span>
                            )}
                          </div>
                        </motion.li>
                      );
                    })}
                  </motion.ul>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {/* Footer */}
      <motion.div
        className="sidebar-footer"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <div className="footer-stat">
          <span className="stat-label">Model</span>
          <span className="stat-value">Local + GPT-5.2</span>
        </div>
        <div className="footer-stat">
          <span className="stat-label">Mode</span>
          <span className="stat-value">Socratic Teaching</span>
        </div>
      </motion.div>
    </aside>
  );
}
