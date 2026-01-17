# LLM Instructor Frontend

A sophisticated, premium learning platform for CS336 Assignment 1. Features a Socratic AI tutor that guides students through implementing transformers from scratch.

## Design Philosophy

**Premium Learning Studio** - Editorial typography meets modern IDE aesthetics:
- Warm amber/copper accents (not typical blue/purple)
- Editorial serif (Crimson Pro) + technical mono (JetBrains Mono)
- Magazine-inspired asymmetric layout
- Refined, professional animations
- Sophisticated dark theme optimized for coding

## Features

- **Real-time Chat**: WebSocket-based streaming for instant AI responses
- **Progress Tracking**: Visual sidebar showing all 12 Assignment 1 functions with status
- **Markdown Rendering**: Beautiful code blocks and formatting for educational content
- **Socratic Teaching**: AI asks questions and provides progressive hints
- **Responsive Design**: Works on desktop and tablet

## Tech Stack

- React 18 + TypeScript
- Vite (fast build tool)
- Framer Motion (smooth animations)
- React Markdown + remark-gfm (markdown rendering)
- WebSocket (real-time communication)

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Backend server running on `localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production

```bash
npm run build
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── Chat.tsx           # Main chat interface
│   ├── Chat.css
│   ├── ProgressSidebar.tsx # Function progress tracking
│   └── ProgressSidebar.css
├── hooks/
│   └── useWebSocket.ts    # WebSocket connection hook
├── types/
│   └── index.ts           # TypeScript interfaces
├── styles/
│   └── globals.css        # Design system & CSS variables
├── App.tsx                # Main application
└── main.tsx               # Entry point
```

## Design System

### Colors
- Primary Background: `#0a0a0a`
- Accent: Warm amber/copper `#d4a574`
- Status Colors: Green (passed), Red (failed), Blue (in progress)

### Typography
- Display/Body: Crimson Pro (serif)
- Code: JetBrains Mono (monospace)

### Spacing
Uses a consistent scale: `xs, sm, md, lg, xl, 2xl, 3xl`

## Connecting to Backend

The frontend expects the backend to be running on `localhost:8000` with:
- WebSocket endpoint: `ws://localhost:8000/ws/chat/{sessionId}`
- HTTP API: `/api/*` endpoints

Update `vite.config.ts` proxy settings if your backend runs on a different port.

## Future Enhancements

- [ ] Code diff viewer
- [ ] Concept quiz mode
- [ ] Export study notes
- [ ] Voice input
- [ ] Mobile optimization

## License

MIT
