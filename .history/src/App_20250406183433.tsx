import React, { useState } from 'react';
import WordExplorer from '../src/components/WordExplorer';
import TestPage from '../src/components/TestPage';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import ErrorBoundary from '../src/components/ErrorBoundary';
import './App.css';

const App: React.FC = () => {
  const [showTestPage, setShowTestPage] = useState<boolean>(false);

  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="app">
          <div className="app-header">
            <button 
              onClick={() => setShowTestPage(!showTestPage)}
              className="toggle-button"
            >
              {showTestPage ? 'Show Main App' : 'Show Test Page'}
            </button>
          </div>
          
          {showTestPage ? (
            <TestPage />
          ) : (
            <WordExplorer />
          )}
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default App;
