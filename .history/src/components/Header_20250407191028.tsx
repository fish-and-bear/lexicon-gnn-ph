import React from 'react';

interface HeaderProps {
  title: string;
  onRandomWord: () => void;
  onTestApiConnection: () => void;
  apiConnected: boolean | null;
}

const Header: React.FC<HeaderProps> = ({ 
  title,
  onRandomWord,
  onTestApiConnection,
  apiConnected
}) => {
  return (
    <header className="header-content">
      <h1>{title}</h1>
      <div className="header-controls">
        <button 
          onClick={onRandomWord} 
          className="random-word-button"
        >
          Random Word
        </button>
        <button 
          onClick={onTestApiConnection} 
          className={`api-test-button ${apiConnected === true ? 'connected' : apiConnected === false ? 'disconnected' : ''}`}
        >
          {apiConnected === true ? 'API Connected' : apiConnected === false ? 'API Disconnected' : 'Test Connection'}
        </button>
      </div>
    </header>
  );
};

export default Header; 