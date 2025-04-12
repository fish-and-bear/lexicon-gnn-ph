import React, { useState, useCallback } from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import InputBase from '@mui/material/InputBase';
import IconButton from '@mui/material/IconButton';
import SearchIcon from '@mui/icons-material/Search';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ArrowForwardIcon from '@mui/icons-material/ArrowForward';
import { styled, alpha } from '@mui/material/styles';
import './Header.css';

// Styled components for the search bar
const Search = styled('div')(({ theme }) => ({
  position: 'relative',
  borderRadius: theme.shape.borderRadius,
  backgroundColor: alpha(theme.palette.common.white, 0.15),
  '&:hover': {
    backgroundColor: alpha(theme.palette.common.white, 0.25),
  },
  marginRight: theme.spacing(2),
  marginLeft: 0,
  width: '100%',
  [theme.breakpoints.up('sm')]: {
    marginLeft: theme.spacing(3),
    width: 'auto',
  },
}));

const SearchIconWrapper = styled('div')(({ theme }) => ({
  padding: theme.spacing(0, 2),
  height: '100%',
  position: 'absolute',
  pointerEvents: 'none',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const StyledInputBase = styled(InputBase)(({ theme }) => ({
  color: 'inherit',
  '& .MuiInputBase-input': {
    padding: theme.spacing(1, 1, 1, 0),
    // vertical padding + font size from searchIcon
    paddingLeft: `calc(1em + ${theme.spacing(4)})`,
    transition: theme.transitions.create('width'),
    width: '100%',
    [theme.breakpoints.up('md')]: {
      width: '30ch',
    },
  },
}));

interface HeaderProps {
  title: string;
  initialValue: string;
  onSearch: (term: string) => void;
  onBack: () => void;
  onForward: () => void;
  canGoBack: boolean;
  canGoForward: boolean;
}

const Header: React.FC<HeaderProps> = ({ 
  title, 
  initialValue, 
  onSearch, 
  onBack, 
  onForward, 
  canGoBack, 
  canGoForward 
}) => {
  const [inputValue, setInputValue] = useState(initialValue);

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  };

  const handleSearchSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSearch(inputValue.trim());
  };

  // Update input value if initialValue changes (e.g., due to history navigation)
  React.useEffect(() => {
    setInputValue(initialValue);
  }, [initialValue]);

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1, display: { xs: 'none', sm: 'block' } }}>
          {title}
        </Typography>
        
        {/* Navigation Buttons */}
        <IconButton 
          color="inherit" 
          onClick={onBack} 
          disabled={!canGoBack}
          aria-label="Go back"
        >
          <ArrowBackIcon />
        </IconButton>
        <IconButton 
          color="inherit" 
          onClick={onForward} 
          disabled={!canGoForward}
          aria-label="Go forward"
          sx={{ mr: 1 }} // Add some margin before search
        >
          <ArrowForwardIcon />
        </IconButton>

        {/* Search Bar */}
        <Search>
          <SearchIconWrapper>
            <SearchIcon />
          </SearchIconWrapper>
          <form onSubmit={handleSearchSubmit}>
            <StyledInputBase
              placeholder="Search word…"
              inputProps={{ 'aria-label': 'search' }}
              value={inputValue}
              onChange={handleInputChange}
            />
          </form>
        </Search>
      </Toolbar>
    </AppBar>
  );
};

export default Header; 