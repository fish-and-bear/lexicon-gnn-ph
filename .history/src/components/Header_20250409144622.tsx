import React from 'react';
import './Header.css';

interface HeaderProps {
  title: string;
  onToggleTheme: () => void;
  onTestApiConnection: () => void;
  onResetCircuitBreaker: () => void;
  apiConnected: boolean | null;
}

const Header: React.FC<HeaderProps> = ({
  title,
  onToggleTheme,
  onTestApiConnection,
  onResetCircuitBreaker,
  apiConnected
}) => {
  return (
    <header className="header-content">
      <h1>{title}</h1>
      <div className="header-buttons">
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
               apiConnected ? '✅ Connected' : '❌ Disconnected'}
        </div>
        <button
          onClick={onToggleTheme}
          className="theme-toggle"
          aria-label="Toggle theme"
        >
          🌙 / ☀️
        </button>
      </div>
    </header>
  );
};

export default Header; 