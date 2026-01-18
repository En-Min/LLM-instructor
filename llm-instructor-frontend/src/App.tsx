import { useState, useEffect } from 'react';
import { Chat } from './components/Chat';
import { ProgressSidebar } from './components/ProgressSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import type { AssignmentPhase } from './types';
import './App.css';

// Complete Assignment 1 function list (21 total functions)
const mockPhases: AssignmentPhase[] = [
  {
    name: 'Phase 1: Basic Operations',
    functions: [
      {
        name: 'linear',
        displayName: 'run_linear',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'embedding',
        displayName: 'run_embedding',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'silu',
        displayName: 'run_silu',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'softmax',
        displayName: 'run_softmax',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'rmsnorm',
        displayName: 'run_rmsnorm',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'cross_entropy',
        displayName: 'run_cross_entropy',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
  {
    name: 'Phase 2: Feed-Forward Networks',
    functions: [
      {
        name: 'swiglu',
        displayName: 'run_swiglu',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
  {
    name: 'Phase 3: Attention Mechanisms',
    functions: [
      {
        name: 'scaled_dot_product_attention',
        displayName: 'run_scaled_dot_product_attention',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'multihead_self_attention',
        displayName: 'run_multihead_self_attention',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'rope',
        displayName: 'run_rope',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'multihead_self_attention_with_rope',
        displayName: 'run_multihead_self_attention_with_rope',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
  {
    name: 'Phase 4: Complete Transformer',
    functions: [
      {
        name: 'transformer_block',
        displayName: 'run_transformer_block',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'transformer_lm',
        displayName: 'run_transformer_lm',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
  {
    name: 'Phase 5: Training & Optimization',
    functions: [
      {
        name: 'get_batch',
        displayName: 'run_get_batch',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'gradient_clipping',
        displayName: 'run_gradient_clipping',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'adamw',
        displayName: 'get_adamw_cls',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'lr_schedule',
        displayName: 'run_get_lr_cosine_schedule',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
  {
    name: 'Phase 6: Checkpointing & Tokenization',
    functions: [
      {
        name: 'save_checkpoint',
        displayName: 'run_save_checkpoint',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'load_checkpoint',
        displayName: 'run_load_checkpoint',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'tokenizer',
        displayName: 'get_tokenizer',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
      {
        name: 'train_bpe',
        displayName: 'run_train_bpe',
        status: 'not_started',
        state: 'concept_check',
        attempts: 0,
        hintsGiven: 0,
      },
    ],
  },
];

function App() {
  const [sessionId] = useState(1); // TODO: Create session via API
  const [phases, _setPhases] = useState<AssignmentPhase[]>(mockPhases);
  const [currentFunction, setCurrentFunction] = useState<string>();

  const { messages, sendMessage, isConnected, isStreaming } = useWebSocket(sessionId);

  // Calculate progress stats
  const totalFunctions = phases.reduce((sum, phase) => sum + phase.functions.length, 0);
  const completedFunctions = phases.reduce(
    (sum, phase) => sum + phase.functions.filter(f => f.status === 'passed').length,
    0
  );

  // TODO: Fetch progress from API
  useEffect(() => {
    // fetch(`/api/progress/${sessionId}`)
    //   .then(res => res.json())
    //   .then(data => setPhases(data));
  }, [sessionId]);

  const handleFunctionSelect = (functionName: string) => {
    setCurrentFunction(functionName);
    sendMessage(`I want to work on ${functionName}`);
  };

  return (
    <div className="app">
      <ProgressSidebar
        phases={phases}
        currentFunction={currentFunction}
        onFunctionSelect={handleFunctionSelect}
      />
      <Chat
        messages={messages}
        onSendMessage={sendMessage}
        onFunctionSelect={handleFunctionSelect}
        isConnected={isConnected}
        isStreaming={isStreaming}
        totalFunctions={totalFunctions}
        completedFunctions={completedFunctions}
      />
    </div>
  );
}

export default App;
