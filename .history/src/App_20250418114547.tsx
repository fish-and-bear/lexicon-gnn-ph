import React from 'react';
import WordExplorer from './components/WordExplorer';
// import ErrorBoundary from './components/ErrorBoundary'; // Keep imported but commented out
import './App.css';

const App: React.FC = () => {
  console.log('App component mounting');
  return (
    // <ErrorBoundary> // Temporarily remove ErrorBoundary
      <div className="app">
        <WordExplorer />
      </div>
    // </ErrorBoundary>
  );
};

export default App;
