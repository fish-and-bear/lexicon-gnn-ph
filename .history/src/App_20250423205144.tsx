import React from 'react';
import WordExplorer from '../src/components/WordExplorer';
// import { ThemeProvider } from '../src/contexts/ThemeContext'; // Old provider
import { AppThemeProvider } from '../src/contexts/ThemeContext'; // New provider
import { Analytics } from '@vercel/analytics/react';
// import ErrorBoundary from '../src/components/ErrorBoundary'; // Temporarily comment out
import './App.css';

const App: React.FC = () => {
  return (
    // <ErrorBoundary> // Temporarily comment out
      // <ThemeProvider> // Old provider
      <AppThemeProvider> { /* New provider */ }
        {/* Remove the redundant app div, CssBaseline handles body styles */}
        {/* <div className="app"> */}
          <WordExplorer />
          <Analytics />
        {/* </div> */}
      </AppThemeProvider>
    // </ErrorBoundary> // Temporarily comment out
  );
};

export default App;
