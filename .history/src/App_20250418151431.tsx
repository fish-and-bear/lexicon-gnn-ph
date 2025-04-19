import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import ErrorBoundary from '../src/components/ErrorBoundary';
import './App.css';

// Basic functional component
const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="app">
          {/* <WordExplorer /> */}
          <p>App component rendered with ThemeProvider and ErrorBoundary.</p>
          {/* <p>ErrorBoundary is currently removed.</p> */}
          <p>WordExplorer is currently commented out.</p>
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default App;
