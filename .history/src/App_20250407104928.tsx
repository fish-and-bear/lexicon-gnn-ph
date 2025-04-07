import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import ErrorBoundary from '../src/components/ErrorBoundary';
import { BrowserRouter } from 'react-router-dom';
import './App.css';

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <ThemeProvider>
          <div className="app">
            <WordExplorer />
          </div>
        </ThemeProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
};

export default App;
