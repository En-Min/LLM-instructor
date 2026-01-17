export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  llmUsed?: 'local' | 'gpt5.2' | 'gemini';
}

export type FunctionStatus = 'not_started' | 'in_progress' | 'passed' | 'failed';

export type TeachingState = 'concept_check' | 'implementation' | 'debug' | 'verify' | 'complete';

export interface FunctionProgress {
  name: string;
  displayName: string;
  status: FunctionStatus;
  state: TeachingState;
  attempts: number;
  hintsGiven: number;
  lastError?: string;
}

export interface AssignmentPhase {
  name: string;
  functions: FunctionProgress[];
}

export interface WebSocketMessage {
  type: 'assistant_chunk' | 'assistant_end' | 'status_update' | 'error';
  content?: string;
  functionName?: string;
  status?: FunctionStatus;
}

export interface SessionState {
  sessionId: number;
  currentFunction?: string;
  messages: Message[];
  progress: AssignmentPhase[];
}
