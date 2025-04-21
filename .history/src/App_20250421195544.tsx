import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import { Analytics } from '@vercel/analytics/react';
// import ErrorBoundary from '../src/components/ErrorBoundary'; // Temporarily comment out
import './App.css';

const App: React.FC = () => {
  return (
    // <ErrorBoundary> // Temporarily comment out
      <ThemeProvider>
        <div className="app">
          <WordExplorer />
          <Analytics />
        </div>
      </ThemeProvider>
    // </ErrorBoundary> // Temporarily comment out
  );
};

export default App;
