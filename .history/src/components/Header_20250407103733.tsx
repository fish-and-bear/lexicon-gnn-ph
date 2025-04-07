import React, { useEffect, useState } from 'react';
import './Header.css';
import { useTheme } from '../contexts/ThemeContext';
import { resetCircuitBreaker, testApiConnection } from '../api/wordApi';

interface HeaderProps {
  title?: string;
  onRandomWord?: () => void;
}

const Header: React.FC<HeaderProps> = ({ title = 'Filipino Lexical Explorer', onRandomWord }) => {
  const { theme, toggleTheme } = useTheme();
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  useEffect(() => {
    const checkApi = async () => {
      try {
        setApiStatus('checking');
        const isConnected = await testApiConnection();
        setApiStatus(isConnected ? 'connected' : 'disconnected');
      } catch (error) {
        setApiStatus('disconnected');
      }
    };

    checkApi();
    // Check API status every 5 minutes
    const intervalId = setInterval(checkApi, 5 * 60 * 1000);
    return () => clearInterval(intervalId);
  }, []);

  const handleApiReset = () => {
    resetCircuitBreaker();
    setApiStatus('checking');
    testApiConnection().then(connected => {
      setApiStatus(connected ? 'connected' : 'disconnected');
    });
  };

  return (
    <header className="header-content">
      <div className="header-left">
        <h1>{title}</h1>
        <div className="header-nav">
          <button
            onClick={() => window.history.back()}
            className="nav-button"
            aria-label="Go back"
            title="Go back"
          >
            ←
          </button>
          <button
            onClick={() => window.history.forward()}
            className="nav-button"
            aria-label="Go forward"
            title="Go forward"
          >
            →
          </button>
        </div>
      </div>
      
      <div className="header-buttons">
        {onRandomWord && (
          <button
            onClick={onRandomWord}
            className="random-button"
            aria-label="Random word"
            title="Get a random word"
          >
            <span className="button-text">Random Word</span> 🎲
          </button>
        )}
          
        <button
          onClick={handleApiReset}
          className="api-button"
          aria-label="Reset API connection"
          title="Reset API connection"
        >
          <span className={`api-status ${apiStatus}`}>
            {apiStatus === 'checking' && '🔄'}
            {apiStatus === 'connected' && '✓'}
            {apiStatus === 'disconnected' && '✗'}
            <span className="button-text">
              {apiStatus === 'checking' && 'Checking...'}
              {apiStatus === 'connected' && 'Connected'}
              {apiStatus === 'disconnected' && 'Disconnected'}
            </span>
          </span>
        </button>
          
        <button
          onClick={toggleTheme}
          className="theme-toggle"
          aria-label="Toggle theme"
          title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
      </div>
    </header>
  );
};

export default Header; 