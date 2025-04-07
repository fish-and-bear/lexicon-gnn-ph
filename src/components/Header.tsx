import React from 'react';
import './Header.css';
import { useTheme } from '../contexts/ThemeContext';
import { Tooltip, IconButton } from '@mui/material';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faRandom, faSun, faMoon, faServer } from '@fortawesome/free-solid-svg-icons';

interface HeaderProps {
  title?: string;
  onRandomWord?: () => void;
  onTestApiConnection?: () => void;
  apiConnected?: boolean | null;
}

const Header: React.FC<HeaderProps> = ({ 
  title = 'Filipino Lexical Explorer', 
  onRandomWord, 
  onTestApiConnection, 
  apiConnected 
}) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="header-content">
      <div className="header-left">
        <h1 className="header-title">{title}</h1>
      </div>
      
      <div className="header-right">
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