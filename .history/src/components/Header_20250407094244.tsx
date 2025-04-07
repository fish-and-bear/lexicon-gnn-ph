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
          🎲 Random
        </button>
        
        <div className={`api-status ${
          apiStatus === null ? 'checking' : 
          apiStatus ? 'connected' : 'disconnected'
        }`}>
          API: {apiStatus === null ? 'Checking...' : 
               apiStatus ? '✅ Connected' : '❌ Disconnected'}
        </div>

        <button
          onClick={onTestApi}
          className="api-button"
          title="Test API connection"
        >
          🔌 Test API
        </button>
        
        <button
          onClick={toggleTheme}
          className="theme-toggle"
          aria-label="Toggle theme"
        >
          {theme === "light" ? "🌙" : "☀️"}
        </button>
      </div>
    </header>
  );
};

export default Header; 