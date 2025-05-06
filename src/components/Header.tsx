import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useAppTheme } from '../contexts/ThemeContext';
import './Header.css';

// MUI imports
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import IconButton from '@mui/material/IconButton';
import Box from '@mui/material/Box';
import Tooltip from '@mui/material/Tooltip';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';
import { styled } from '@mui/material/styles';

interface HeaderProps {
  title?: string;
  onTestApiConnection?: () => void;
  onResetCircuitBreaker?: () => void;
  apiConnected?: boolean | null;
}

const StyledAppBar = styled(AppBar)(({ theme }) => ({
  backgroundColor: 'var(--primary-color)',
  boxShadow: 'none',
  borderBottom: '1px solid var(--card-border-color)',
  '.dark &': {
    backgroundColor: 'var(--header-color)',
  }
}));

const ThemeToggle = styled(IconButton)(({ theme }) => ({
  color: '#ffffff',
  marginLeft: theme.spacing(1),
  '.dark &': {
    color: 'var(--text-color-white)',
  }
}));

const Header: React.FC<HeaderProps> = ({
  title = "Fil-Relex: Filipino Root Word Explorer",
  onTestApiConnection = () => {},
  onResetCircuitBreaker = () => {},
  apiConnected = null
}) => {
  const { themeMode, toggleTheme } = useAppTheme();
  const location = useLocation();
  const [baybayinMenuAnchor, setBaybayinMenuAnchor] = React.useState<null | HTMLElement>(null);

  const handleBaybayinMenuOpen = (event: React.MouseEvent<HTMLButtonElement>) => {
    setBaybayinMenuAnchor(event.currentTarget);
  };

  const handleBaybayinMenuClose = () => {
    setBaybayinMenuAnchor(null);
  };

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <StyledAppBar position="static">
      <Toolbar sx={{ 
        justifyContent: { xs: 'center', sm: 'space-between' }, 
        flexWrap: 'wrap'
      }}>
        <Box sx={{ display: { xs: 'block', sm: 'block' } }}>
          <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
            {title}
          </Typography>
        </Box>

        <Box sx={{ 
          display: 'flex', 
          alignItems: 'center',
          flexBasis: { xs: '100%', sm: 'auto' },
          justifyContent: { xs: 'center', sm: 'flex-end' },
          mt: { xs: 1, sm: 0 }
        }}>
          <Link 
            to="/" 
            className={`header-link ${isActive('/') ? 'active' : ''}`}
          >
            Word Explorer
          </Link>

          <Button
            onClick={handleBaybayinMenuOpen}
            color="inherit"
            sx={{ mx: 0.5 }}
          >
            Baybayin Tools
          </Button>
          <Menu
            anchorEl={baybayinMenuAnchor}
            open={Boolean(baybayinMenuAnchor)}
            onClose={handleBaybayinMenuClose}
            className="baybayin-menu"
          >
            <Link 
              to="/baybayin/converter" 
              className="header-link" 
              onClick={handleBaybayinMenuClose}
              style={{ color: 'inherit', display: 'block', padding: '8px 16px' }}
            >
              <MenuItem selected={isActive('/baybayin/converter')}>
                Baybayin Converter
              </MenuItem>
            </Link>
            <Link 
              to="/baybayin/search" 
              className="header-link" 
              onClick={handleBaybayinMenuClose}
              style={{ color: 'inherit', display: 'block', padding: '8px 16px' }}
            >
              <MenuItem selected={isActive('/baybayin/search')}>
                Baybayin Search
              </MenuItem>
            </Link>
            <Link 
              to="/baybayin/statistics" 
              className="header-link" 
              onClick={handleBaybayinMenuClose}
              style={{ color: 'inherit', display: 'block', padding: '8px 16px' }}
            >
              <MenuItem selected={isActive('/baybayin/statistics')}>
                Baybayin Statistics
              </MenuItem>
            </Link>
          </Menu>

          {onTestApiConnection && onResetCircuitBreaker && (
            <Box sx={{ mx: 1, display: 'flex', alignItems: 'center' }}>
              <Tooltip title="Test API connection">
                <IconButton 
                  onClick={onTestApiConnection} 
                  color="inherit" 
                  size="small" 
                  sx={{ mx: 0.5 }}
                >
                  🔌
                </IconButton>
              </Tooltip>
              <Tooltip title="Reset API circuit breaker">
                <IconButton 
                  onClick={onResetCircuitBreaker} 
                  color="inherit" 
                  size="small" 
                  sx={{ mx: 0.5 }}
                >
                  🔄
                </IconButton>
              </Tooltip>
              {apiConnected !== null && (
                <Typography variant="caption" sx={{ 
                  mx: 0.5, 
                  color: apiConnected ? '#4caf50' : '#f44336',
                  backgroundColor: 'rgba(0,0,0,0.1)',
                  px: 1,
                  py: 0.5,
                  borderRadius: 1
                }}>
                  {apiConnected ? '✅ API Connected' : '❌ API Disconnected'}
                </Typography>
              )}
            </Box>
          )}

          <Tooltip title="Toggle light/dark theme">
            <ThemeToggle onClick={toggleTheme} color="inherit" size="small">
              {themeMode === 'dark' ? '☀️' : '🌙'}
            </ThemeToggle>
          </Tooltip>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
};

export default Header; 