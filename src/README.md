# Frontend Application

A modern React TypeScript application providing an interactive interface for exploring the Filipino Lexical Resource database with advanced visualization and search capabilities.

## 🏗️ Architecture Overview

The frontend is built with:
- **React 18** - Modern React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Vite** - Fast build tool and development server
- **Material-UI (MUI)** - Consistent UI components
- **D3.js** - Interactive data visualizations
- **React Query** - Server state management

## 📁 Project Structure

```
src/
├── components/                # Reusable React components
│   ├── WordGraph.tsx         # Interactive word relationship visualization
│   ├── WordDetails.tsx       # Word information display
│   ├── WordExplorer.tsx      # Main application container
│   └── NetworkControls.tsx   # Graph control interface
├── contexts/                 # React contexts
│   └── ThemeContext.tsx      # Dark/light theme management
├── api/                      # API integration layer
│   └── wordApi.ts           # Backend API client functions
├── utils/                    # Utility functions
│   ├── colorUtils.ts        # Color mapping for visualizations
│   └── caching.ts           # Client-side caching
├── styles/                   # CSS and styling
│   └── global.css           # Global styles and CSS variables
├── types.ts                  # TypeScript type definitions
├── App.tsx                   # Main application component
└── index.tsx                # Application entry point
```

## 🚀 Quick Start

### Prerequisites

- Node.js 16 or higher
- npm or yarn
- Modern web browser with ES2020 support

### Installation

1. **Clone and Navigate:**
   ```bash
   git clone <repository-url>
   cd fil-relex
   ```

2. **Install Dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Environment Configuration:**
   Create `.env` file in the root directory:
   ```bash
   # Backend API Configuration
   VITE_API_BASE_URL=http://localhost:5000/api/v2
   
   # Application Configuration
   VITE_VERSION=1.0.0
   VITE_APP_NAME=FilRelex
   
   # Optional: Feature Flags
   VITE_ENABLE_DEBUG=false
   VITE_ENABLE_ANALYTICS=false
   ```

4. **Start Development Server:**
   ```bash
   npm run dev
   # or
   yarn dev
   ```

The application will be available at `http://localhost:5173`

## 🎨 Component Architecture

### Core Components

#### WordExplorer (`components/WordExplorer.tsx`)
The main application container that manages:
- Search functionality and word selection
- Navigation history and breadcrumbs
- Panel layout and responsive design
- State management for word data

#### WordGraph (`components/WordGraph.tsx`)
Interactive visualization component featuring:
- D3.js-powered network graph rendering
- Dynamic node positioning and relationship display
- Interactive filtering and exploration tools
- Responsive design for mobile and desktop

#### WordDetails (`components/WordDetails.tsx`)
Comprehensive word information display:
- Definitions and etymology
- Part-of-speech information
- Baybayin script representation
- Related words and relationships

#### NetworkControls (`components/NetworkControls.tsx`)
Graph control interface providing:
- Depth and breadth adjustment sliders
- Relationship type filtering
- Legend and visualization controls
- Mobile-optimized interaction

### Context Providers

#### ThemeContext (`contexts/ThemeContext.tsx`)
Manages application-wide theming:
- Dark/light mode toggle
- System preference detection
- Persistent theme storage
- CSS variable updates

## 🎯 Key Features

### Interactive Word Network Visualization
- **Dynamic Graph Rendering**: Real-time visualization of word relationships
- **Interactive Exploration**: Click, hover, and navigation interactions
- **Filtering Options**: Filter by relationship types and connection depth
- **Responsive Design**: Optimized for both desktop and mobile devices

### Advanced Search Capabilities
- **Autocomplete**: Real-time word suggestions as you type
- **Fuzzy Matching**: Find words even with partial or approximate spelling
- **Multi-language Support**: Search across Filipino and related languages
- **History Navigation**: Browse through previously viewed words

### Baybayin Script Integration
- **Native Script Display**: Traditional Filipino script representation
- **Transliteration**: Automatic conversion between scripts
- **Cultural Context**: Educational information about script usage

### Performance Optimizations
- **Smart Caching**: Client-side caching of API responses
- **Virtual Scrolling**: Efficient rendering of large datasets
- **Code Splitting**: Lazy loading of components and routes
- **Optimistic Updates**: Immediate UI feedback

## 🛠️ Development

### Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Preview production build
npm run preview

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format

# Run tests
npm run test

# Run tests with coverage
npm run test:coverage
```

### Code Style Guidelines

#### TypeScript Best Practices
```typescript
// Use explicit interface definitions
interface WordData {
  id: number;
  lemma: string;
  definitions: Definition[];
}

// Prefer function components with proper typing
const WordCard: React.FC<WordCardProps> = ({ word, onSelect }) => {
  // Component implementation
};

// Use custom hooks for complex logic
const useWordSearch = (query: string) => {
  const [results, setResults] = useState<SearchResult[]>([]);
  // Hook implementation
};
```

#### Component Structure
```tsx
// Component file structure
import React, { useState, useCallback } from 'react';
import { SomeExternalLibrary } from 'external-library';
import { InternalComponent } from '../internal/Component';
import './ComponentName.css';

interface ComponentProps {
  // Props definition
}

const ComponentName: React.FC<ComponentProps> = ({ prop1, prop2 }) => {
  // State and effects
  const [state, setState] = useState();
  
  // Callbacks
  const handleAction = useCallback(() => {
    // Implementation
  }, [dependencies]);
  
  // Render
  return (
    <div className="component-name">
      {/* JSX content */}
    </div>
  );
};

export default ComponentName;
```

### State Management

#### Local State
```typescript
// Use useState for component-local state
const [searchQuery, setSearchQuery] = useState<string>('');
const [isLoading, setIsLoading] = useState<boolean>(false);
```

#### Global State
```typescript
// Use Context for shared state
const { themeMode, toggleTheme } = useAppTheme();
```

#### Server State
```typescript
// Use React Query for server state
const { data: wordData, isLoading, error } = useQuery(
  ['word', wordId],
  () => fetchWordDetails(wordId)
);
```

## 🎨 Styling and Theming

### CSS Architecture

The application uses a combination of:
- **CSS Variables** for dynamic theming
- **CSS Modules** for component-specific styles
- **Material-UI** for consistent component styling

### Theme System

```css
/* Light theme variables */
:root {
  --primary-color: #1d3557;
  --secondary-color: #e63946;
  --accent-color: #f1faee;
  --background-color: #f8f9fa;
  --text-color: #1d3557;
  --card-bg-color: #ffffff;
}

/* Dark theme variables */
[data-theme="dark"] {
  --primary-color: #ffd166;
  --secondary-color: #e63946;
  --accent-color: #2a9d8f;
  --background-color: #0a0d16;
  --text-color: #e0e0e0;
  --card-bg-color: #131826;
}
```

### Responsive Design

```css
/* Mobile-first approach */
.component {
  /* Mobile styles */
}

@media (min-width: 768px) {
  .component {
    /* Tablet styles */
  }
}

@media (min-width: 1024px) {
  .component {
    /* Desktop styles */
  }
}
```

## 🚀 Deployment

### Production Build

```bash
# Create optimized production build
npm run build

# Preview the production build locally
npm run preview
```

### Static Hosting Options

#### Vercel (Recommended)
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
vercel --prod
```

Configuration (`vercel.json`):
```json
{
  "framework": "vite",
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

#### Netlify
```bash
# Build command: npm run build
# Publish directory: dist
```

Configuration (`netlify.toml`):
```toml
[build]
  command = "npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

#### AWS S3 + CloudFront
```bash
# Build the application
npm run build

# Upload to S3 bucket
aws s3 sync dist/ s3://your-bucket-name --delete

# Invalidate CloudFront cache
aws cloudfront create-invalidation --distribution-id YOUR_DISTRIBUTION_ID --paths "/*"
```

### Environment Variables for Production

```bash
# Production API endpoint
VITE_API_BASE_URL=https://api.filrelex.com/api/v2

# Production configuration
VITE_VERSION=1.0.0
VITE_ENABLE_DEBUG=false
VITE_ENABLE_ANALYTICS=true

# Analytics configuration (if enabled)
VITE_GA_TRACKING_ID=GA_MEASUREMENT_ID
```

## 📱 Mobile Optimization

### Responsive Features
- **Touch-Optimized Interactions**: Tap targets sized for mobile use
- **Swipe Gestures**: Navigation through word relationships
- **Adaptive Layout**: Panel resizing and stacking for small screens
- **Performance Tuning**: Optimized rendering for mobile devices

### PWA Capabilities
```typescript
// Service worker registration
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}

// App manifest for installable PWA
// See public/manifest.json
```

## 🔍 Testing

### Unit Testing
```bash
# Run Jest tests
npm run test

# Run tests with coverage
npm run test:coverage

# Run tests in watch mode
npm run test:watch
```

### Component Testing
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { WordCard } from './WordCard';

test('displays word information correctly', () => {
  const mockWord = { id: 1, lemma: 'test', definitions: [] };
  render(<WordCard word={mockWord} onSelect={jest.fn()} />);
  
  expect(screen.getByText('test')).toBeInTheDocument();
});
```

### E2E Testing
```bash
# Install Playwright
npm install -D @playwright/test

# Run E2E tests
npx playwright test
```

## 🔧 Performance Optimization

### Bundle Analysis
```bash
# Analyze bundle size
npm run build -- --analyze

# Visualize bundle composition
npx vite-bundle-analyzer
```

### Optimization Techniques
- **Code Splitting**: Automatic route-based splitting
- **Tree Shaking**: Eliminates unused code
- **Asset Optimization**: Image compression and format optimization
- **Caching Strategies**: Browser and service worker caching

## 📞 Support

### Common Issues

**Build Errors:**
- Clear node_modules and reinstall: `rm -rf node_modules package-lock.json && npm install`
- Check Node.js version compatibility

**API Connection Issues:**
- Verify VITE_API_BASE_URL is correct
- Check CORS configuration on backend
- Ensure backend server is running

**Styling Issues:**
- Clear browser cache
- Check CSS variable definitions
- Verify theme context is properly wrapped

### Development Tools

**Recommended VS Code Extensions:**
- ES7+ React/Redux/React-Native snippets
- TypeScript Importer
- Prettier - Code formatter
- ESLint
- Auto Rename Tag

**Browser DevTools:**
- React Developer Tools
- Redux DevTools (if using Redux)
- Network tab for API debugging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](../LICENSE.md) file for details.

---

For more information, see the main project [README.md](../README.md) 