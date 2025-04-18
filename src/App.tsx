import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
import { ThemeProvider } from '../src/contexts/ThemeContext';
// import ErrorBoundary from '../src/components/ErrorBoundary'; // Temporarily comment out
import './App.css';

const App: React.FC = () => {
  return (
    // <ErrorBoundary> // Temporarily comment out
      <ThemeProvider>
        <div className="app">
          <WordExplorer />
        </div>
      </ThemeProvider>
    // </ErrorBoundary> // Temporarily comment out
  );
};

export default App;
