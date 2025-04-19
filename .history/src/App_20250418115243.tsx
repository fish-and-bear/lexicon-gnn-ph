import React from 'react';
import WordExplorer from './components/WordExplorer'; // Restore this
// import ErrorBoundary from './components/ErrorBoundary'; // Keep commented out for now
import './App.css';

const App: React.FC = () => {
  console.log('App component mounting');
  return (
    // <ErrorBoundary>
      <div className="app">
        <WordExplorer /> {/* Restore rendering WordExplorer */}
        {/* <div>App Rendered - No WordExplorer</div> */}
      </div>
    // </ErrorBoundary>
  );
};

export default App;
