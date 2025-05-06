// Minimal bootstrap file with careful module loading order
console.log('[BOOTSTRAP] Application bootstrap starting');

// Step 1: Import core React dependencies only
import React from 'react';
import { createRoot } from 'react-dom/client';

// Step 2: Basic CSS only for initial rendering
import './styles/global.css';

// Step 3: Import App component directly
import App from './App';
import './index.css';
import reportWebVitals from './reportWebVitals';
import { AppThemeProvider, useAppTheme } from './contexts/ThemeContext';
import { createTheme, ThemeProvider as MuiThemeProvider, CssBaseline } from '@mui/material';

// Wrapper component to integrate custom theme with MUI theme
const AppWithMuiTheme: React.FC = () => {
  const { themeMode } = useAppTheme();

  // Define explicit dark mode background colors
  const darkPalette = {
    mode: 'dark' as 'dark',
    background: {
      default: '#0a0d16', // From WordExplorer.css
      paper: '#131826',   // From WordExplorer.css
    },
    text: {
      primary: '#e0e0e0', // From WordExplorer.css
      secondary: '#a0a0a0', // From WordExplorer.css
    },
    primary: {
      main: '#ffd166', // CORRECTED: Yellowish from WordExplorer.css
    },
    secondary: {
      main: '#e63946', // Red from WordExplorer.css
    },
    // Add other palette colors if needed, matching WordExplorer.css
    // e.g., button colors if not handled by component overrides
  };

  // Define explicit light mode background colors
  const lightPalette = {
    mode: 'light' as 'light',
    background: {
      default: '#f8f9fa',
      paper: '#ffffff',
    },
    text: {
      primary: '#1d3557',
      secondary: '#6c757d',
    },
    primary: {
      main: '#1d3557',
    },
    secondary: {
      main: '#e63946', // Match old_src_2 secondary
    },
    // ... other light palette settings ...
  };

  // Create MUI theme based on the custom context's mode
  const muiTheme = React.useMemo(
    () => createTheme({
      palette: themeMode === 'dark' ? darkPalette : lightPalette,
      components: {
        // Add slider styling overrides
        MuiSlider: {
          styleOverrides: {
            root: ({ ownerState, theme }) => ({
              // Use theme palette for color
              color: theme.palette.primary.main,
              height: 3,
            }),
            thumb: ({ ownerState, theme }) => ({
              height: 12,
              width: 12,
              // Use theme palette for color
              backgroundColor: theme.palette.primary.main,
              '&:hover, &.Mui-focusVisible': {
                boxShadow: 'none',
              },
            }),
            track: ({ ownerState, theme }) => ({
              height: 3,
              border: 'none',
              // Use accent color for light mode track, primary for dark mode track
              backgroundColor: theme.palette.mode === 'light' ? 'var(--accent-color)' : theme.palette.primary.main,
            }),
            rail: ({ ownerState, theme }) => ({
              height: 3,
              opacity: 1,
              backgroundColor: theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.1)' : 'rgba(255, 255, 255, 0.15)',
            }),
            mark: ({ ownerState, theme }) => ({
              height: 4,
              width: 1,
              marginTop: -1,
              backgroundColor: theme.palette.mode === 'light' ? 'rgba(0, 0, 0, 0.2)' : 'rgba(255, 255, 255, 0.3)',
            }),
            markActive: ({ ownerState, theme }) => ({
              opacity: 1,
              // Use accent for light, primary for dark
              backgroundColor: theme.palette.mode === 'light' ? 'var(--accent-color)' : theme.palette.primary.main,
            }),
            valueLabel: ({ ownerState, theme }) => ({
              fontSize: '0.7rem',
              padding: '2px 6px',
              borderRadius: '4px',
              // Use primary color for background
              backgroundColor: theme.palette.primary.main,
              // Use button text color variable (assuming it's defined globally)
              color: theme.palette.mode === 'light' ? 'var(--button-text-color)' : '#0a0d16', // Dark mode button text is dark
            }),
          },
        },
        // Ensure buttons use the right colors from CSS vars or palette
        MuiButton: {
          styleOverrides: {
            root: {
              textTransform: 'none',
              fontSize: '0.875rem',
            },
            // Style contained buttons using CSS variables directly
            contained: {
              backgroundColor: 'var(--button-color)',
              color: 'var(--button-text-color)',
              '&:hover': {
                 // Simple filter adjustment for hover
                 filter: 'brightness(1.1)',
              },
              '&.Mui-disabled': {
                 // Use MUI's default disabled styles or customize if needed
              }
            },
          },
        },
      },
    }),
    [themeMode]
  );

  return (
    <MuiThemeProvider theme={muiTheme}>
      <CssBaseline />
      <App />
    </MuiThemeProvider>
  );
};

// Add error handling for React initialization
try {
  console.log("[INIT] index.tsx initializing...");
  
  // Get the root element
  const rootElement = document.getElementById('root');
  
  if (!rootElement) {
    throw new Error('Failed to find the root element');
  }
  
  // Create a root
  const root = createRoot(rootElement);
  
  // Render the app with proper providers
  root.render(
    <React.StrictMode>
      <AppThemeProvider>
        <AppWithMuiTheme />
      </AppThemeProvider>
    </React.StrictMode>
  );
  
  console.log("[INIT] App successfully rendered");
} catch (error) {
  console.error('[FATAL] Failed to initialize React application:', error);
  // Display error to user
  const rootElement = document.getElementById('root');
  if (rootElement) {
    rootElement.innerHTML = `
      <div style="color: #721c24; background-color: #f8d7da; padding: 20px; border-radius: 5px;">
        <h2>Application Error</h2>
        <p>The application failed to initialize: ${error instanceof Error ? error.message : 'Unknown error'}</p>
        <button onclick="window.location.reload()" style="margin-top: 10px; padding: 8px 16px; background-color: #721c24; color: white; border: none; border-radius: 4px; cursor: pointer;">
          Reload Application
        </button>
      </div>
    `;
  }
}

// Disable automatic performance tracking to simplify bootstrapping
// This can be re-enabled later if needed
// import reportWebVitals from './reportWebVitals';
// reportWebVitals();