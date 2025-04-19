import React from 'react';
import WordExplorer from './components/WordExplorer';
import ErrorBoundary from './components/ErrorBoundary';
import './App.css';

const App: React.FC = () => {
  console.log('App component mounting');
  return (
    <ErrorBoundary>
      <div className="app">
        <WordExplorer />
      </div>
    </ErrorBoundary>
  );
};

export default App;
