import React from 'react';
import './Header.css';
import SearchBar from './SearchBar';

interface HeaderProps {
  title: string;
  onSearch: (query: string) => void;
  theme: string;
  toggleTheme: () => void;
  onGoBack: () => void;
  onGoForward: () => void;
  canGoBack: boolean;
  canGoForward: boolean;
  onRandomWord: () => void;
  onTestApi: () => void;
  apiStatus: boolean | null;
  children?: React.ReactNode;
}

const Header: React.FC<HeaderProps> = ({
  title,
  onSearch,
  theme,
  toggleTheme,
  onGoBack,
  onGoForward,
  canGoBack,
  canGoForward,
  onRandomWord,
  onTestApi,
  apiStatus,
  children
}) => {
  return (
    <header className="header-content">
      <div className="header-left">
        <h1>{title}</h1>
        <div className="header-nav">
          <button 
            onClick={onGoBack} 
            disabled={!canGoBack}
            className="nav-button"
            title="Go back"
          >
            ←
          </button>
          <button 
            onClick={onGoForward} 
            disabled={!canGoForward}
            className="nav-button"
            title="Go forward"
          >
            →
          </button>
        </div>
      </div>
      
      <div className="header-center">
        {children}
      </div>
      
      <div className="header-buttons">
        <button
          onClick={onRandomWord}
          className="random-button"
          title="Explore a random word"
        >
          <span className="button-icon">🎲</span>
          <span className="button-text">Random</span>
        </button>
        
        <div className={`api-status ${
          apiStatus === null ? 'checking' : 
          apiStatus ? 'connected' : 'disconnected'
        }`}>
          {apiStatus === null ? '⏳ Checking...' : 
           apiStatus ? '✅ Connected' : '❌ Disconnected'}
        </div>

        <button
          onClick={onTestApi}
          className="api-button"
          title="Test API connection"
        >
          <span className="button-icon">🔌</span>
          <span className="button-text">Test</span>
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