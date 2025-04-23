import React, { createContext, useContext, useState, ReactNode, useMemo, useEffect } from 'react';
import { ThemeProvider as MuiThemeProvider, createTheme, CssBaseline, PaletteMode } from '@mui/material';

// Define the types for your context
interface AppThemeContextType {
  themeMode: PaletteMode;
  toggleTheme: () => void;
}

// Create the context with default values
const AppThemeContext = createContext<AppThemeContextType | undefined>(undefined);

export const AppThemeProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // Determine initial theme from localStorage or system preference
  const getInitialTheme = (): PaletteMode => {
    try {
      const storedTheme = localStorage.getItem('appTheme') as PaletteMode;
      if (storedTheme === 'light' || storedTheme === 'dark') {
        return storedTheme;
      }
    } catch (e) {
      console.error("Error reading theme from localStorage", e);
    }
    // Fallback to system preference
    // const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    // return prefersDark ? 'dark' : 'light'; // Temporarily default to light
    return 'light'; // Default to light for now
  };

  const [themeMode, setThemeMode] = useState<PaletteMode>(getInitialTheme);

  // Update localStorage when theme changes
  useEffect(() => {
    try {
      localStorage.setItem('appTheme', themeMode);
      // Apply class to body for CSS variable scoping
      document.body.classList.remove('light', 'dark');
      document.body.classList.add(themeMode);
    } catch (e) {
      console.error("Error saving theme to localStorage", e);
    }
  }, [themeMode]);

  const toggleTheme = () => {
    setThemeMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  // Create the MUI theme object based on the current mode
  const muiTheme = useMemo(() => createTheme({
    palette: {
      mode: themeMode,
      // You can customize MUI's default light/dark palettes here if needed
      // For example:
      // primary: {
      //   main: themeMode === 'light' ? '#1d3557' : '#ffd166', // Match your CSS var --primary-color
      // },
      // background: {
      //    default: themeMode === 'light' ? '#ffffff' : '#0a0d16', // Match --bg-color
      //    paper: themeMode === 'light' ? '#ffffff' : '#131826', // Match --card-bg
      // }
      // ... other customizations
    },
    // You can also customize components, typography, etc.
    components: {
      // Example: Ensure Paper background respects theme
      MuiPaper: {
        styleOverrides: {
          root: ({ theme }) => ({
            backgroundColor: theme.palette.background.paper, // Use MUI theme background
            color: theme.palette.text.primary, // Use MUI theme text color
          }),
        },
      },
      MuiButton: {
        styleOverrides: {
          // Ensure contained buttons use theme defaults unless overridden by sx
          contained: ({ theme }) => ({
            // Example: Set default contained button colors if needed
            // backgroundColor: theme.palette.primary.main,
            // color: theme.palette.primary.contrastText,
          }),
        }
      }
      // ... other component overrides
    }
  }), [themeMode]);

  return (
    <AppThemeContext.Provider value={{ themeMode, toggleTheme }}>
      {/* Apply MUI Theme & Reset/Baseline */}
      <MuiThemeProvider theme={muiTheme}>
        <CssBaseline /> {/* Normalize styles and apply background color */} 
        {/* Apply light/dark class to a wrapper if needed, but CssBaseline handles body bg */}
        {/* <div className={themeMode}>{children}</div> */}
        {children} 
      </MuiThemeProvider>
    </AppThemeContext.Provider>
  );
};

// Custom hook to use the AppThemeContext
export const useAppTheme = () => {
  const context = useContext(AppThemeContext);
  if (!context) {
    throw new Error('useAppTheme must be used within an AppThemeProvider');
  }
  return context;
};

// Export the context for direct access if needed
export { AppThemeContext };

// Remove the old ThemeToggleButton as theme toggle is likely handled elsewhere (e.g., Header)
/*
export const ThemeToggleButton: React.FC = () => {
  const { themeMode, toggleTheme } = useAppTheme();

  return (
    <button onClick={toggleTheme}>
      Switch to {themeMode === 'light' ? 'dark' : 'light'} mode
    </button>
  );
};
*/
