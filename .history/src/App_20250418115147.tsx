import React from 'react';
// import WordExplorer from './components/WordExplorer'; // Keep imported, don't render
// import ErrorBoundary from './components/ErrorBoundary'; // Keep commented out
import './App.css';

const App: React.FC = () => {
  console.log('App component mounting');
  return (
    // <ErrorBoundary>
      <div className="app">
        {/* <WordExplorer /> */} {/* Commented out WordExplorer */}
        <div>App Rendered - No WordExplorer</div>
      </div>
    // </ErrorBoundary>
  );
};

export default App;
