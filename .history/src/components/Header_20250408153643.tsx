import React from 'react';

export interface HeaderProps {
  title?: string;
  theme?: string;
  onToggleTheme?: () => void;
  onTestApiConnection: () => Promise<void> | void;
  onResetCircuitBreaker?: () => void;
  apiConnected: boolean | null;
}

const Header: React.FC<HeaderProps> = ({ 
  title = "Filipino Lexical Explorer",
  theme,
  onToggleTheme,
  onTestApiConnection,
  onResetCircuitBreaker,
  apiConnected
}) => {
  return (
    <header className="header-content">
      <h1>{title}</h1>
      <div className="header-controls">
        {onToggleTheme && (
          <button 
            onClick={onToggleTheme} 
            className="theme-toggle"
          >
            {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
          </button>
        )}
        <button 
          onClick={onTestApiConnection} 
          className={`api-test-button ${apiConnected === true ? 'connected' : apiConnected === false ? 'disconnected' : ''}`}
        >
          {apiConnected === true ? 'API Connected' : apiConnected === false ? 'API Disconnected' : 'Test Connection'}
        </button>
        {onResetCircuitBreaker && (
          <button 
            onClick={onResetCircuitBreaker} 
            className="reset-circuit-button"
          >
            Reset API Connection
          </button>
        )}
      </div>
    </header>
  );
};

export default Header; 