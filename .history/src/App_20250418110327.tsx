import React from 'react';
import WordExplorer from './components/WordExplorer';
import { ThemeProvider } from './contexts/ThemeContext';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

const App: React.FC = () => {
  console.log('App component mounting');
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <div className="app">
          <WordExplorer />
        </div>
      </ThemeProvider>
    </ErrorBoundary>
  );
};

export default App;
