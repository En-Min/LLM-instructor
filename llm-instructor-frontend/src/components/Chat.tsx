import { useRef, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Message } from '../types';
import { WelcomeScreen } from './WelcomeScreen';
import './Chat.css';

interface ChatProps {
  messages: Message[];
  onSendMessage: (content: string) => void;
  onFunctionSelect: (functionName: string) => void;
  isConnected: boolean;
  isStreaming: boolean;
  totalFunctions: number;
  completedFunctions: number;
}

export function Chat({ messages, onSendMessage, onFunctionSelect, isConnected, isStreaming, totalFunctions, completedFunctions }: ChatProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${inputRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim() && !isStreaming) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleSubmit(e);
    }
  };

  return (
    <div className="chat-container">
      {/* Header */}
      <header className="chat-header">
        <div className="header-content">
          <h1 className="chat-title">LLM Instructor</h1>
          <div className="connection-status">
            <span className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
            <span className="status-text">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      {/* Messages or Welcome Screen */}
      <div className="messages-container">
        {messages.length === 0 ? (
          <WelcomeScreen
            onStartLearning={onFunctionSelect}
            totalFunctions={totalFunctions}
            completedFunctions={completedFunctions}
          />
        ) : (
        <>
        <AnimatePresence initial={false}>
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              className={`message ${message.role}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
            >
              <div className="message-header">
                <span className="message-role">
                  {message.role === 'user' ? 'You' : 'Instructor'}
                </span>
                {message.llmUsed && (
                  <span className="message-model">{message.llmUsed}</span>
                )}
              </div>
              <div className="message-content">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code: ({ className, children, ...props }) => {
                      const isInline = !className;
                      return isInline ? (
                        <code className="inline-code" {...props}>
                          {children}
                        </code>
                      ) : (
                        <pre className="code-block">
                          <code className={className} {...props}>
                            {children}
                          </code>
                        </pre>
                      );
                    },
                    p: ({ children }) => <p className="markdown-p">{children}</p>,
                    ul: ({ children }) => <ul className="markdown-ul">{children}</ul>,
                    ol: ({ children }) => <ol className="markdown-ol">{children}</ol>,
                    li: ({ children }) => <li className="markdown-li">{children}</li>,
                    blockquote: ({ children }) => (
                      <blockquote className="markdown-blockquote">{children}</blockquote>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isStreaming && (
          <motion.div
            className="streaming-indicator"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <span className="dot" style={{ animationDelay: '0s' }} />
            <span className="dot" style={{ animationDelay: '0.2s' }} />
            <span className="dot" style={{ animationDelay: '0.4s' }} />
          </motion.div>
        )}

        <div ref={messagesEndRef} />
        </>
        )}
      </div>

      {/* Input */}
      <form className="chat-input-container" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Ask a question or tell me which function you want to work on..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={!isConnected || isStreaming}
            rows={1}
          />
          <button
            type="submit"
            className="send-button"
            disabled={!input.trim() || !isConnected || isStreaming}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path
                d="M2 10L18 2L10 18L8 11L2 10Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                fill="none"
              />
            </svg>
          </button>
        </div>
        <p className="input-hint">
          Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to send
        </p>
      </form>
    </div>
  );
}
