import React, { useEffect, useState } from 'react';
import './Header.css';
import { useTheme } from '../contexts/ThemeContext';
import { resetCircuitBreaker, testApiConnection } from '../api/wordApi';
import { Button, Tooltip, IconButton } from '@mui/material';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRandom, faSun, faMoon, faServer } from '@fortawesome/free-solid-svg-icons';

interface HeaderProps {
  title?: string;
  onRandomWord?: () => void;
  onTestApiConnection?: () => void;
  apiConnected?: boolean | null;
}

const Header: React.FC<HeaderProps> = ({ title = 'Filipino Lexical Explorer', onRandomWord, onTestApiConnection, apiConnected }) => {
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
        {onTestApiConnection && (
          <Tooltip title="Test API Connection" arrow>
            <IconButton 
              onClick={onTestApiConnection} 
              className={`api-test-button ${apiConnected === true ? 'connected' : apiConnected === false ? 'disconnected' : ''}`}
              aria-label="Test API Connection"
            >
              <FontAwesomeIcon icon={faServer} />
            </IconButton>
          </Tooltip>
        )}
        {onRandomWord && (
          <Tooltip title="Random Word" arrow>
            <IconButton 
              onClick={onRandomWord} 
              className="random-button"
              aria-label="Get Random Word"
            >
              <FontAwesomeIcon icon={faRandom} />
            </IconButton>
          </Tooltip>
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
          
        <Tooltip title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`} arrow>
          <IconButton 
            onClick={toggleTheme} 
            className="theme-toggle"
            aria-label={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
          >
            <FontAwesomeIcon icon={theme === 'dark' ? faSun : faMoon} />
          </IconButton>
        </Tooltip>
      </div>
    </header>
  );
};

export default Header; 