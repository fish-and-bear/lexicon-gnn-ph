import React, { useState, useCallback } from 'react';
import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import WordExplorer from './components/WordExplorer';
import { BaybayinConverter, BaybayinSearch, BaybayinStatistics } from './components';
import Header from './components/Header';
import { testApiConnection, resetCircuitBreaker } from './api/wordApi';

// Simple error boundary component
class ErrorBoundary extends React.Component<{children: React.ReactNode}, {hasError: boolean, error: Error | null}> {
  constructor(props: {children: React.ReactNode}) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("Error caught by App ErrorBoundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ 
          padding: '20px', 
          color: '#721c24',
          backgroundColor: '#f8d7da',
          borderRadius: '5px',
          margin: '20px',
          fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
          <h2>Application Error</h2>
          <p>{this.state.error?.message || "An unknown error occurred"}</p>
          <button 
            onClick={() => window.location.reload()} 
            style={{
              marginTop: '10px',
              padding: '8px 16px',
              backgroundColor: '#721c24',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Create a simple application component
const App: React.FC = () => {
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);

  const handleTestApiConnection = useCallback(async () => {
    try {
      const result = await testApiConnection();
      setApiConnected(result);
    } catch (err) {
      console.error("Error testing API connection:", err);
      setApiConnected(false);
    }
  }, []);

  const handleResetCircuitBreaker = useCallback(() => {
    try {
      resetCircuitBreaker();
      console.log("Circuit breaker reset successful");
      // Test connection after reset
      handleTestApiConnection();
    } catch (err) {
      console.error("Error resetting circuit breaker:", err);
    }
  }, [handleTestApiConnection]);

  // Test API connection on initial load
  React.useEffect(() => {
    handleTestApiConnection();
  }, [handleTestApiConnection]);

  return (
    <ErrorBoundary>
      <Router>
        <div className="app">
          <Header 
            title="Fil-Relex: Filipino Root Explorer"
            onTestApiConnection={handleTestApiConnection}
            onResetCircuitBreaker={handleResetCircuitBreaker}
            apiConnected={apiConnected}
          />
          <Routes>
            <Route path="/" element={<WordExplorer />} />
            <Route path="/baybayin/converter" element={<BaybayinConverter />} />
            <Route path="/baybayin/search" element={<BaybayinSearch />} />
            <Route path="/baybayin/statistics" element={<BaybayinStatistics />} />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
};

export default App;
