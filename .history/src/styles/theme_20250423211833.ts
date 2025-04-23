import { PaletteMode } from '@mui/material';

// Define common tokens if needed (e.g., typography, spacing - simplified for now)
const typography = {
  fontFamily: [
    '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"Helvetica Neue"',
    'Arial', 'sans-serif', '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"',
  ].join(','),
};

const spacing = (factor: number) => `${0.5 * factor}rem`; // Example spacing unit

// Define Color Palettes (Ensure these match the CSS variables we added previously)
export const lightThemeColors = {
  background: '#ffffff',
  text: '#212529',
  textSecondary: '#6c757d',
  primary: '#0d6efd', // Example Blue
  primaryRGB: '13, 110, 253',
  secondary: '#6c757d', // Example Gray
  accent: '#fd7e14', // Example Orange
  accentRGB: '253, 126, 20',
  cardBg: '#ffffff',
  cardBgElevated: '#f8f9fa',
  cardBgLight: 'rgba(0, 0, 0, 0.03)',
  cardBorder: '#dee2e6',
  inputBg: '#ffffff',
  inputBorder: '#ced4da',
  button: '#0d6efd',
  buttonText: '#ffffff',
  shadow: 'rgba(0, 0, 0, 0.1)',
  // Graph Colors
  graphMain: '#5d9cec',
  graphRoot: '#ff7088',
  graphSynonym: '#64b5f6',
  graphAntonym: '#5c6bc0',
  graphDerived: '#4dd0e1',
  graphVariant: '#9575cd',
  graphRelated: '#4fc3f7',
  graphAssociated: '#90a4ae',
  graphEtymology: '#d00000',
  graphDerivative: '#606c38',
  graphDefault: '#78909c',
};

export const darkThemeColors = {
  background: '#121212',
  text: '#e0e0e0',
  textSecondary: '#a0a0a0',
  primary: '#4dabf7', // Lighter Blue
  primaryRGB: '77, 171, 247',
  secondary: '#adb5bd', // Lighter Gray
  accent: '#ff922b', // Lighter Orange
  accentRGB: '255, 146, 43',
  cardBg: '#1e1e1e',
  cardBgElevated: '#2a2a2a',
  cardBgLight: 'rgba(255, 255, 255, 0.05)',
  cardBorder: '#424242',
  inputBg: '#2a2a2a',
  inputBorder: '#555555',
  button: '#4dabf7',
  buttonText: '#121212',
  shadow: 'rgba(255, 255, 255, 0.1)',
  // Graph Colors
  graphMain: '#6ba7ff',
  graphRoot: '#ff8ba0',
  graphSynonym: '#7cc1ff',
  graphAntonym: '#8e99f3',
  graphDerived: '#6ffbff',
  graphVariant: '#b39ddb',
  graphRelated: '#77d3ff',
  graphAssociated: '#b0bec5',
  graphEtymology: '#ff5252',
  graphDerivative: '#a5d6a7',
  graphDefault: '#a0a0a0',
};

// Function to generate MUI palette configuration
export const getThemePalette = (mode: PaletteMode) => ({
  mode,
  ...(mode === 'light'
    ? {
        primary: { main: lightThemeColors.primary },
        secondary: { main: lightThemeColors.secondary },
        background: { default: lightThemeColors.background, paper: lightThemeColors.cardBg },
        text: { primary: lightThemeColors.text, secondary: lightThemeColors.textSecondary },
        divider: lightThemeColors.cardBorder,
        // You can add more MUI palette properties here, mapping from your tokens
        accent: { main: lightThemeColors.accent }, // Example custom color
      }
    : {
        primary: { main: darkThemeColors.primary },
        secondary: { main: darkThemeColors.secondary },
        background: { default: darkThemeColors.background, paper: darkThemeColors.cardBg },
        text: { primary: darkThemeColors.text, secondary: darkThemeColors.textSecondary },
        divider: darkThemeColors.cardBorder,
        // You can add more MUI palette properties here, mapping from your tokens
        accent: { main: darkThemeColors.accent }, // Example custom color
      }),
  // Common palette properties can go here if they don't change with mode
});

// Extend MUI's Palette interface if adding custom colors like 'accent'
// declare module '@mui/material/styles' {
//   interface Palette {
//     accent?: Palette['primary'];
//   }
//   interface PaletteOptions {
//     accent?: PaletteOptions['primary'];
//   }
// }
// Extend Button color prop if needed
// declare module '@mui/material/Button' {
//   interface ButtonPropsColorOverrides {
//     accent: true;
//   }
// }

// Function to generate CSS variables string (optional, if needed for non-MUI parts)
const generateCssVariables = (colors: typeof lightThemeColors | typeof darkThemeColors): string => {
  return Object.entries(colors)
    .map(([key, value]) => `--${key.replace(/([A-Z])/g, '-$1').toLowerCase()}: ${value};`)
    .join('\n  ');
};

// console.log('Light CSS Vars:\n', generateCssVariables(lightThemeColors));
// console.log('Dark CSS Vars:\n', generateCssVariables(darkThemeColors)); 