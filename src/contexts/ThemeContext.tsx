import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Add logging
console.log("[INIT] ThemeContext.tsx loading");

// Define the shape of our theme context
interface ThemeContextType {
  themeMode: 'light' | 'dark';
  toggleTheme: () => void;
}

// Create the context with a default value
const ThemeContext = createContext<ThemeContextType>({
  themeMode: 'light',
  toggleTheme: () => console.warn('No theme provider found')
});

// Hook to use the theme context
export const useAppTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useAppTheme must be used within a AppThemeProvider');
  }
  return context;
};

// Props for our theme provider
interface AppThemeProviderProps {
  children: ReactNode;
}

// Theme provider component
export const AppThemeProvider: React.FC<AppThemeProviderProps> = ({ children }) => {
  // Get the preferred theme from localStorage or system preference
  const getInitialTheme = (): 'light' | 'dark' => {
    // Check if user has previously set a preference in localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'light' || savedTheme === 'dark') {
      return savedTheme;
    }
    
    // Check system preference using media query
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    // Default to light theme
    return 'light';
  };

  // Set up state for theme
  const [themeMode, setThemeMode] = useState<'light' | 'dark'>(getInitialTheme);

  // Toggle theme function
  const toggleTheme = () => {
    setThemeMode(prevMode => {
      const newMode = prevMode === 'light' ? 'dark' : 'light';
      localStorage.setItem('theme', newMode);
      return newMode;
    });
  };

  // Apply theme class to body
  useEffect(() => {
    document.body.classList.remove('light', 'dark');
    document.body.classList.add(themeMode);
  }, [themeMode]);

  // Effect for updating theme when system preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      // Only update if user hasn't set a preference in localStorage
      if (!localStorage.getItem('theme')) {
        setThemeMode(mediaQuery.matches ? 'dark' : 'light');
      }
    };

    // Add event listener
    mediaQuery.addEventListener('change', handleChange);

    // Clean up
    return () => {
      mediaQuery.removeEventListener('change', handleChange);
    };
  }, []);

  // Provide the theme context
  return (
    <ThemeContext.Provider value={{ themeMode, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeContext;

console.log("[INIT] ThemeContext.tsx loaded successfully");
