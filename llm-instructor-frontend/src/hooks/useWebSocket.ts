import { useState, useEffect, useCallback, useRef } from 'react';
import type { Message, WebSocketMessage } from '../types';

export function useWebSocket(sessionId: number) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const streamingMessageRef = useRef<string>('');

  useEffect(() => {
    // Connect to WebSocket backend
    const ws = new WebSocket(`ws://localhost:8000/ws/chat/${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected to backend');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);
        console.log('[WS] Received:', data.type, data.content?.substring(0, 20));

        if (data.type === 'assistant_chunk') {
          setIsStreaming(true);
          streamingMessageRef.current += data.content || '';
          console.log('[WS] Accumulated:', streamingMessageRef.current.length, 'chars');

          // Update the last message or create new one
          const newContent = streamingMessageRef.current;
          setMessages(prev => {
            const lastMessage = prev[prev.length - 1];
            console.log('[WS] Updating messages, last role:', lastMessage?.role, 'content length:', newContent.length);
            if (lastMessage && lastMessage.role === 'assistant' && lastMessage.id === 'streaming') {
              const updated = [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  content: newContent,
                }
              ];
              console.log('[WS] Updated existing message, total messages:', updated.length);
              return updated;
            } else {
              const updated: Message[] = [
                ...prev,
                {
                  id: 'streaming',
                  role: 'assistant' as const,
                  content: newContent,
                  timestamp: new Date(),
                  llmUsed: 'local',
                }
              ];
              console.log('[WS] Created new message, total messages:', updated.length);
              return updated;
            }
          });
        } else if (data.type === 'assistant_end') {
          setIsStreaming(false);

          // Finalize the streaming message
          setMessages(prev => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage && lastMessage.id === 'streaming') {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  id: crypto.randomUUID(),
                }
              ];
            }
            return prev;
          });

          streamingMessageRef.current = '';
        }
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = useCallback((content: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.error('WebSocket not connected');
      return;
    }

    // Add user message immediately
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);

    // Send to backend
    wsRef.current.send(JSON.stringify({ content }));
  }, []);

  return {
    messages,
    sendMessage,
    isConnected,
    isStreaming,
  };
}
