import React from 'react';
import './common.css';
import './Header.css';

interface HeaderProps {
  title: string;
  onToggleTheme: () => void;
  onTestApiConnection: () => void;
  onResetCircuitBreaker: () => void;
  apiConnected: boolean | null;
  onRandomClick?: () => void;
  isRandomLoading?: boolean;
  isLoading?: boolean;
  theme?: string;
}

const Header: React.FC<HeaderProps> = ({
  title,
  onToggleTheme,
  onTestApiConnection,
  onResetCircuitBreaker,
  apiConnected,
  onRandomClick,
  isRandomLoading = false,
  isLoading = false,
  theme = 'light'
}) => {
  return (
    <header className="header-content">
      <h1>{title}</h1>
      <div className="header-buttons">
        {onRandomClick && (
          <button
            onClick={onRandomClick}
            className="random-button"
            title="Get a random word"
            disabled={isRandomLoading || isLoading}
          >
            {isRandomLoading ? '⏳ Loading...' : '🎲 Random Word'}
          </button>
        )}
        <button
          onClick={onResetCircuitBreaker}
          className="debug-button"
          title="Reset API connection"
        >
          🔄 Reset API
        </button>
        <button
          onClick={onTestApiConnection}
          className="debug-button"
          title="Test API connection"
        >
          🔌 Test API
        </button>
        <div className={`api-status ${
          apiConnected === null ? 'checking' : 
          apiConnected ? 'connected' : 'disconnected'
        }`}>
          API: {apiConnected === null ? 'Checking...' : 
               apiConnected ? '✓ Connected' : '✗ Disconnected'}
        </div>
        <button
          onClick={onToggleTheme}
          className="theme-toggle"
          aria-label="Toggle theme"
        >
          {theme === 'light' ? '🌙' : '☀️'}
        </button>
      </div>
    </header>
  );
};

export default Header; 