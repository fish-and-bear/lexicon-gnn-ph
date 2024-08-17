import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
import { ThemeProvider } from '../src/contexts/ThemeContext';
import './App.css';

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <div className="app">
        <WordExplorer />
      </div>
    </ThemeProvider>
  );
};

export default App;
