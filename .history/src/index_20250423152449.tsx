import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles/global.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';
import { createTheme, ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material';

// --- BEGIN DIAGNOSTIC --- 
// Moved after imports to fix eslint rule
try {
  localStorage.setItem('__test', '1');
  localStorage.removeItem('__test');
  console.log('[DIAG] localStorage access OK.');
} catch (e) {
  console.error('[DIAG] localStorage access FAILED:', e);
  // Optional: Display a message to the user if critical
  // alert('Error accessing localStorage. App may not function correctly.');
}
// --- END DIAGNOSTIC --- 

// Clear the circuit breaker state on application start to ensure a fresh start
try {
  // Only clear in development mode to avoid clearing in production unnecessarily
  if (process.env.NODE_ENV === 'development') {
    resetCircuitBreaker();
    console.log('Circuit breaker state cleared on startup');
  }
} catch (e) {
  console.error('Error clearing circuit breaker:', e);
}

// Wrapper component to integrate custom theme with MUI theme
const AppWithMuiTheme: React.FC = () => {
  const { theme: customThemeMode } = useTheme(); // Get 'light' or 'dark'

  // Define explicit dark mode background colors
  const darkPalette = {
    mode: 'dark' as 'dark', // Cast to PaletteMode
    background: {
      default: '#0a0d16', // Match --bg-color from CSS
      paper: '#131826',   // Match --card-bg-color from CSS
    },
    text: {
      primary: '#e0e0e0', // Match --text-color
      secondary: '#a0a0a0', // Match --text-color-secondary
    },
    primary: { // Example: Use the yellow from CSS vars
      main: '#ffd166', // Match --primary-color in .dark
    },
    // Add other palette customizations as needed
  };

  // Define explicit light mode background colors (or use MUI defaults)
  const lightPalette = {
    mode: 'light' as 'light', // Cast to PaletteMode
    background: {
      default: '#f8f9fa', // Match --bg-color
      paper: '#ffffff',   // Match --card-bg-color
    },
     text: {
      primary: '#1d3557', // Match --text-color
      secondary: '#6c757d', // Match --color-default or similar
    },
     primary: { // Example: Use the blue from CSS vars
      main: '#1d3557', // Match --primary-color in :root
    },
    // Add other palette customizations as needed
  };

  // Create MUI theme based on the custom context's mode
  const muiTheme = React.useMemo(
    () =>
      createTheme({
        palette: customThemeMode === 'dark' ? darkPalette : lightPalette,
        // Add other MUI theme options like typography, components overrides etc.
      }),
    [customThemeMode]
  );

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline /> {/* Ensure CssBaseline uses the theme backgrounds */}
      <App />
    </MuiThemeProvider>
  );
};

// Use the new createRoot API
const container = document.getElementById('root');
if (container) { // Ensure container is not null
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <ThemeProvider>
        <AppWithMuiTheme />
      </ThemeProvider>
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element. Check public/index.html");
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();