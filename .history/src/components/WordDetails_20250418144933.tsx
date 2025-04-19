import React, { useCallback, useState, useEffect, useMemo } from 'react';
import { Definition, WordInfo, RelatedWord, NetworkLink, NetworkNode, WordForm, WordTemplate, Idiom, Affixation, DefinitionCategory, DefinitionLink, DefinitionRelation } from '../types';
// import { convertToBaybayin } from '../api/wordApi';
import './common.css'; // Import common CSS first
import './WordDetails.css';
// import './Tabs.css';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Link from '@mui/material/Link';
import { styled, useTheme, alpha } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery'; // Import useMediaQuery
import Drawer from '@mui/material/Drawer';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import RefreshIcon from '@mui/icons-material/Refresh';
import Grid from '@mui/material/Grid';

// Add d3 import here at the top
import * as d3 from 'd3';

// MUI Icons
// import VolumeUpIcon from '@mui/icons-material/VolumeUp';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import StopCircleIcon from '@mui/icons-material/StopCircle'; // Icon for stop button

interface WordDetailsProps {
  wordInfo: WordInfo;
  etymologyTree: any;
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  onWordLinkClick: (word: string) => void;
  onEtymologyNodeClick: (node: any) => void;
}

// After initial imports, add this interface to describe any record
interface AnyRecord {
  [key: string]: any;
}

// Helper function to format relation type names
const formatRelationType = (type: string): string => {
  // Special case for specific types
  if (type === 'kaugnay') return 'Kaugnay';
  if (type === 'kasalungat') return 'Kasalungat';
  if (type === 'kahulugan') return 'Kahulugan';
  
  // Capitalize the first letter and replace underscores with spaces
  return type
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

// Define relation color scheme - aligned with WordGraph component
const relationColors: Record<string, string> = {
  // Core
  main: "#0e4a86",      // Deep blue - standout color for main word
  
  // Origin group - Reds and oranges
  root: "#e63946",      // Bright red
  etymology: "#d00000", // Dark red
  cognate: "#ff5c39",   // Light orange
  
  // Meaning group - Blues
  synonym: "#457b9d",   // Medium blue
  related: "#48cae4",   // Light blue
  antonym: "#023e8a",   // Dark blue
  similar: "#a8dadc",   // Pale blue (similar to related)
  
  // Form group - Purples
  variant: "#7d4fc3",   // Medium purple
  spelling: "#9d4edd",  // Light purple
  abbreviation: "#6247aa", // Dark purple
  form_of: "#6a4c93",   // Blue-purple
  
  // Hierarchy group - Greens
  taxonomic: "#2a9d8f", // Teal
  hypernym: "#2a9d8f",  // Teal (same as taxonomic)
  hyponym: "#52b788",   // Light green
  meronym: "#40916c",   // Forest green
  holonym: "#40916c",   // Forest green (same as meronym)
  part_whole: "#40916c", // Forest green
  component_of: "#40916c", // Forest green
  component: "#40916c", // Forest green
  
  // Derivational group - Yellow-greens
  derived: "#2a9d8f",   // Teal green
  affix: "#588157",     // Olive
  derivative: "#606c38", // Dark olive
  
  // Info group - Yellows/Oranges
  usage: "#fcbf49",     // Gold
  
  // Specific Filipino relations - Oranges and pinks
  kaugnay: "#fb8500",   // Orange
  salita: "#E91E63",    // Pink
  kahulugan: "#c9184a", // Dark pink
  kasalungat: "#e63946", // Red
  
  // Fallback
  associated: "#adb5bd", // Neutral gray
  other: "#6c757d"      // Dark gray
};

// Define key graph colors locally for styling
const graphColors = {
  main: "#5d9cec",     // Brighter blue for dark mode
  root: "#ff7088",     // Lighter red
  synonym: "#64b5f6",  // Brighter blue
  antonym: "#5c6bc0",  // Indigo
  derived: "#4dd0e1",  // Cyan
  variant: "#9575cd",  // Lighter purple
  related: "#4fc3f7",  // Light blue
  associated: "#90a4ae", // Blue-grey
  default: "#78909c"   // Blue-grey
};

// Helper to determine if a background color is light or dark
const isColorLight = (hexColor: string): boolean => {
  try {
    const color = d3.color(hexColor);
    if (!color) return true; // Default to light if parsing fails
    const rgb = color.rgb();
    // Standard luminance calculation
    const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    return luminance > 0.5;
  } catch (e) {
    return true;
  }
};

// --- Styled Components ---
// Simplified Accordion Styling
const StyledAccordion = styled(Accordion)(({ theme }) => ({
  border: 'none', // Remove explicit border
  boxShadow: 'none', // Remove default shadow
  backgroundColor: 'transparent',
  '&:not(:last-child)': {
    borderBottom: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : theme.palette.divider}`, // Enhance dark mode divider
  },
  '&::before': {
    display: 'none', // Remove the default top border pseudo-element
  },
}));

const StyledAccordionSummary = styled(AccordionSummary)(({ theme }) => ({
  padding: theme.spacing(0, 1), // Adjust padding
  minHeight: 48,
  backgroundColor:
    theme.palette.mode === 'dark'
      ? 'rgba(255, 255, 255, .05)'
      : 'rgba(0, 0, 0, .02)',
  '&:hover': {
     backgroundColor:
        theme.palette.mode === 'dark'
          ? 'rgba(255, 255, 255, .1)'
          : 'rgba(0, 0, 0, .03)',
  },
  '& .MuiAccordionSummary-content': {
    margin: theme.spacing(1.5, 0), // Adjust margin
    alignItems: 'center', // Vertically align title and chip
  },
  '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': {
    transform: 'rotate(180deg)',
  },
}));

const StyledAccordionDetails = styled(AccordionDetails)(({ theme }) => ({
  padding: theme.spacing(2, 2, 2, 2), // Consistent padding
  borderTop: 'none', // Remove internal border
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.02)' : 'transparent',
  // Remove any white borders that might be coming from the default component
  border: 'none',
  '& .MuiPaper-root': {
    borderColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.05)' : theme.palette.divider,
  }
}));

const ExpandMoreIcon = () => <Typography sx={{ transform: 'rotate(90deg)', lineHeight: 0, color: 'text.secondary' }}>▶</Typography>;
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'primary.main' }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'error.main' }}>⏹️</Typography>;

// Add a styled component for definitions, sources, etc. that properly handles dark mode
const DefinitionCard = styled(Box)(({ theme }) => ({
  padding: theme.spacing(2),
  marginBottom: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.mode === 'dark' 
    ? 'var(--definition-bg-color)' 
    : theme.palette.background.paper,
  border: `1px solid ${theme.palette.mode === 'dark' 
    ? 'rgba(255, 255, 255, 0.08)' 
    : theme.palette.divider}`,
  boxShadow: theme.palette.mode === 'dark'
    ? '0 2px 4px rgba(0, 0, 0, 0.2)'
    : 'none',
}));

// Styled definition item
const DefinitionItem = styled(Box)(({ theme }) => ({
  marginBottom: theme.spacing(1.5),
  '&:last-child': {
    marginBottom: 0,
  },
}));

// Source tag for definitions
const SourceTag = styled(Typography)(({ theme }) => ({
  fontSize: '0.75rem',
  fontStyle: 'italic',
  marginTop: theme.spacing(0.5),
  color: theme.palette.mode === 'dark' 
    ? 'var(--link-color)'
    : theme.palette.text.secondary,
}));

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  etymologyTree, // Keep props even if tab is commented
  isLoadingEtymology,
  etymologyError,
  onWordLinkClick,
  onEtymologyNodeClick
}) => {
  const theme = useTheme();
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md'));
  const isDarkMode = theme.palette.mode === 'dark';

  const [activeTab, setActiveTab] = useState<string>('definitions'); // Default to definitions
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

  // ... (audio handling useEffect and handleTabChange) ...
  useEffect(() => {
    // Cleanup function to stop audio when component unmounts or audio source changes
    return () => {
      if (audioElement) {
        audioElement.pause();
        audioElement.currentTime = 0;
        setIsAudioPlaying(false);
      }
    };
  }, [audioElement]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setActiveTab(newValue);
    // Stop audio when changing tabs
    if (audioElement) {
      audioElement.pause();
      setIsAudioPlaying(false);
    }
  };

  const playAudio = (url: string) => {
    // ... (implementation as before) ...
  };

  const stopAudio = () => {
    // ... (implementation as before) ...
  };
  
  const renderHeader = () => { /* ... (keep existing implementation) ... */ };
  const renderDefinitionsTab = () => { /* ... (keep existing implementation) ... */ };
  // Comment out other render functions for now
  // const renderRelationsTab = () => { /* ... */ };
  // const renderFormsAndTemplatesTab = () => { /* ... */ };
  // const renderEtymologyTab = () => { /* ... */ };
  // const renderSourcesInfoTab = () => { /* ... */ };

  // Determine available tabs dynamically based on wordInfo content
  const availableTabs = useMemo(() => {
    const tabs = [
      { label: 'Definitions', value: 'definitions', content: renderDefinitionsTab, disabled: !wordInfo?.definitions?.length },
      // { label: 'Etymology', value: 'etymology', content: renderEtymologyTab, disabled: !wordInfo?.etymologies?.length && !etymologyTree }, // Commented out
      // { label: 'Relations', value: 'relations', content: renderRelationsTab, disabled: !wordInfo?.incoming_relations?.length && !wordInfo?.outgoing_relations?.length && !wordInfo?.semantic_network }, // Commented out
      // { label: 'Forms/Templates', value: 'forms', content: renderFormsAndTemplatesTab, disabled: !wordInfo?.forms?.length && !wordInfo?.templates?.length }, // Commented out
      // { label: 'Sources/Meta', value: 'sources', content: renderSourcesInfoTab, disabled: false }, // Commented out
    ];
    return tabs.filter(tab => !tab.disabled);
  }, [wordInfo, etymologyTree]); // Dependencies might need adjustment later

  // Reset tab if current activeTab is no longer available
  useEffect(() => {
    if (availableTabs.length > 0 && !availableTabs.some(tab => tab.value === activeTab)) {
      setActiveTab(availableTabs[0].value);
    }
  }, [availableTabs, activeTab]);

  return (
    <Paper elevation={2} sx={{ 
        display: 'flex', 
        flexDirection: isWideScreen ? 'row' : 'column', 
        height: '100%', // Ensure it takes full height of its container
        overflow: 'hidden', // Prevent internal scrolling issues
        border: `1px solid ${theme.palette.divider}`,
        backgroundColor: isDarkMode ? '#2C2C2C' : '#f9f9f9',
      }}
    >
      {/* Header Section */} 
      <Box sx={{ p: 2, borderBottom: !isWideScreen ? `1px solid ${theme.palette.divider}` : 'none', borderRight: isWideScreen ? `1px solid ${theme.palette.divider}` : 'none' }}>
        {renderHeader()}
      </Box>

      {/* Content Section (Tabs and Details) */} 
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: isWideScreen ? 'row' : 'column', overflow: 'hidden' }}>
        {/* Tabs */} 
        <Tabs
          orientation={isWideScreen ? 'vertical' : 'horizontal'}
          variant="scrollable"
          value={activeTab} // Use string value
          onChange={handleTabChange}
          aria-label="Word details tabs"
          sx={{
            borderRight: isWideScreen ? `1px solid ${theme.palette.divider}` : 'none',
            borderBottom: !isWideScreen ? `1px solid ${theme.palette.divider}` : 'none',
            borderColor: 'divider',
            minWidth: isWideScreen ? 150 : 'auto', // Ensure tabs have minimum width vertically
            flexShrink: 0,
            bgcolor: isDarkMode ? '#3a3a3a' : '#f0f0f0',
            '& .MuiTabs-indicator': {
              backgroundColor: theme.palette.primary.main,
            }
          }}
        >
          {availableTabs.map((tab) => (
            <Tab key={tab.value} label={tab.label} value={tab.value} disabled={tab.disabled} />
          ))}
        </Tabs>

        {/* Tab Content */} 
        <Box sx={{ p: isWideScreen ? 3 : 2, flexGrow: 1, overflowY: 'auto', backgroundColor: isDarkMode ? '#222' : '#fff' }}>
          {availableTabs.map((tab) => (
            <div key={tab.value} role="tabpanel" hidden={activeTab !== tab.value}>
              {activeTab === tab.value && tab.content()}
            </div>
          ))}
        </Box>
      </Box>
    </Paper>
  );
});

export default WordDetails;