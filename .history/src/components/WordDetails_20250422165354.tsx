import React, { useCallback, useState, useEffect, useRef } from 'react';
import { Definition, WordInfo, WordForm, WordTemplate, Idiom, Affixation, Credit, BasicWord, EtymologyTree, WordSuggestion } from '../types'; // Added EtymologyTree and WordSuggestion
// import { convertToBaybayin } from '../api/wordApi';
import './WordDetails.css';
// Import color utility functions needed
import { getNodeColor, getTextColorForBackground } from '../utils/colorUtils'; 
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
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Link from '@mui/material/Link';
import { styled, useTheme, alpha, Theme } from '@mui/material/styles'; // Import Theme type
import useMediaQuery from '@mui/material/useMediaQuery'; // Import useMediaQuery
import Button from '@mui/material/Button';

// MUI Icons
// import VolumeUpIcon from '@mui/icons-material/VolumeUp';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import StopCircleIcon from '@mui/icons-material/StopCircle'; // Icon for stop button

interface WordDetailsProps {
  wordData: WordInfo; // Use wordData
  etymologyTree: EtymologyTree | null; // Use EtymologyTree type
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  onFetchEtymology: (wordId: number) => Promise<EtymologyTree | null>; 
  onWordClick: (word: string | WordSuggestion | BasicWord | null) => void; // Use WordSuggestion/BasicWord
  isMobile: boolean; 
  isLoading: boolean; // Add isLoading prop for details
  containerId: string; // Add the containerId prop
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

// Define relation type colors - Enhance colors for better dark mode support
const relationColors: { [key: string]: string } = {
  // Core relations
  main: "#0e4a86",      // Deep blue
  root: "#e63946",      // Bright red
  
  // Meaning group - Blues
  synonym: "#457b9d",   // Medium blue
  antonym: "#023e8a",   // Dark blue
  related: "#48cae4",   // Light blue
  similar: "#4cc9f0",   // Sky blue
  
  // Origin group - Reds and oranges
  etymology: "#d00000", // Dark red
  cognate: "#ff5c39",   // Light orange
  
  // Form group - Purples
  variant: "#7d4fc3",   // Medium purple
  spelling_variant: "#9381ff", // Lavender
  regional_variant: "#b8b8ff", // Light lavender
  
  // Hierarchy group - Greens
  hypernym: "#2a9d8f",  // Teal
  hyponym: "#40916c",   // Forest green
  taxonomic: "#52b788", // Medium green
  meronym: "#74c69d",   // Light green
  holonym: "#95d5b2",   // Pale green
  part_whole: "#52b788", // Medium green
  component: "#74c69d", // Light green
  
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

// --- Styled Components ---
// Simplified Accordion Styling
const StyledAccordion = styled(Accordion)(({ theme }: { theme: Theme }) => ({ // Add Theme type
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

const StyledAccordionSummary = styled(AccordionSummary)(({ theme }: { theme: Theme }) => ({ // Add Theme type
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

const StyledAccordionDetails = styled(AccordionDetails)(({ theme }: { theme: Theme }) => ({ // Add Theme type
  padding: theme.spacing(2, 2, 2, 2), // Consistent padding
  borderTop: 'none', // Remove internal border
  backgroundColor: theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.02)' : 'transparent',
}));

const ExpandMoreIcon = () => <Typography sx={{ transform: 'rotate(90deg)', lineHeight: 0, color: 'text.secondary' }}>▶</Typography>;
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'primary.main' }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'error.main' }}>⏹️</Typography>;

// *** ADD Helper Function to strip leading list numbers/letters ***
const stripLeadingNumber = (text: string): string => {
  if (!text) return '';
  // Regex to match patterns like: 1., 1), a., a), I., I) followed by space
  // It handles digits, lowercase/uppercase letters, and Roman numerals (basic cases)
  const strippedText = text.trim().replace(/^(\d+|[a-zA-Z]+|[IVXLCDM]+) *[\.\)]\s+/, '');
  return strippedText.trim(); // Return the trimmed base text
};

// Define the component using React.forwardRef
const WordDetailsComponent = React.forwardRef<HTMLDivElement, WordDetailsProps>((
  {
    wordData,
    etymologyTree,
    isLoadingEtymology,
    etymologyError,
    onFetchEtymology,
    onWordClick,
    isMobile,
    isLoading,
    containerId
  },
  ref
) => {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const [activeTab, setActiveTab] = useState(0);
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  const [filteredDefinitions, setFilteredDefinitions] = useState<Definition[]>([]);

  // Ref for the scrollable container *within* WordDetails if needed for internal logic,
  // but generally rely on the containerId prop for ResizeObserver.
  const internalScrollRef = useRef<HTMLDivElement>(null);

  // Effect to setup audio element
  useEffect(() => {
    setIsAudioPlaying(false); // Stop previous audio on word change
    const audioPronunciation = wordData?.pronunciations?.find(p => p.type === 'audio' && p.value);
    let audio: HTMLAudioElement | null = null;

    if (audioPronunciation?.value) {
      try {
          audio = new Audio(audioPronunciation.value);
          const onEnded = () => setIsAudioPlaying(false);
          audio.addEventListener('ended', onEnded);
          setAudioElement(audio);

          return () => {
            if (audio) {
              audio.pause();
              audio.removeEventListener('ended', onEnded);
            }
          };
      } catch (error) {
          console.error("Error creating audio element:", error);
          setAudioElement(null); // Ensure state is cleared on error
      }
    } else {
      setAudioElement(null); // Clear if no audio pron
    }
    
    return undefined; // Add default return for useEffect
  }, [wordData]);

  // Effect to filter definitions when wordData changes
  useEffect(() => {
    // ... filtering logic ...
  }, [wordData.definitions]);

  // Effect for ResizeObserver
  useEffect(() => {
      // Find the container using the passed ID
      const scrollContainer = document.getElementById(containerId);
      if (!scrollContainer) {
          console.error("WordDetails: Scroll container not found with ID:", containerId);
          return;
      }

      // --- ResizeObserver Logic ---
      // Check if ResizeObserver is supported
      if (typeof ResizeObserver === 'undefined') {
          console.warn("ResizeObserver not supported by this browser.");
          return;
      }

      const resizeObserver = new ResizeObserver(entries => {
          for (let entry of entries) {
              // Example: Log the container width when it changes
              // console.log('WordDetails container width:', entry.contentRect.width);
              // You could potentially adjust internal layouts based on this
          }
      });

      // Observe the container found by ID
      resizeObserver.observe(scrollContainer);

      // Cleanup function to disconnect the observer
      return () => {
          resizeObserver.unobserve(scrollContainer);
          resizeObserver.disconnect();
      };
  }, [containerId]); // Re-run effect if containerId changes (though unlikely)

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const playAudio = useCallback(() => {
    if (!audioElement) return;
    if (isAudioPlaying) {
      audioElement.pause();
      audioElement.currentTime = 0;
      setIsAudioPlaying(false);
    } else {
      audioElement.play().then(() => setIsAudioPlaying(true)).catch(err => {
        console.error("Audio play failed:", err);
        setIsAudioPlaying(false); // Reset state on error
      });
    }
  }, [audioElement, isAudioPlaying]);

  // --- Rendering Sections ---

  const renderHeader = () => {
    // Make IPA check case-insensitive
    const ipaPronunciation = wordData.pronunciations?.find(p => p.type?.toLowerCase() === 'ipa');
    const hasAudio = wordData.pronunciations?.some(p => p.type === 'audio' && p.value);
    const tags = wordData.tags ? wordData.tags.split(',').map(tag => tag.trim()).filter(Boolean) : [];

    // More elegant background colors based on theme
    const headerBgColor = isDarkMode
      ? alpha(graphColors.main, 0.15) // Subtle blue background in dark mode
      : alpha(graphColors.main, 0.07); // Even more subtle in light mode

    // Get the main color from the theme for accents
    const mainColor = isDarkMode 
      ? theme.palette.primary.light // Use lighter primary in dark mode
      : theme.palette.primary.main;
    
    // Determine text color based on background
    const headerTextColor = isDarkMode 
      ? theme.palette.primary.contrastText // Use contrast text color
      : theme.palette.getContrastText(headerBgColor);

    return (
      // More elegant header with subtle styling
      <Box sx={{ 
        bgcolor: isDarkMode ? 'rgba(30, 40, 60, 0.4)' : headerBgColor, 
        color: headerTextColor, 
        // Reduced padding for mobile
        pt: theme.spacing(isMobile ? 1.5 : 3), 
        pb: theme.spacing(isMobile ? 1 : 1.5), 
        pl: theme.spacing(isMobile ? 1.5 : 3), 
        pr: theme.spacing(isMobile ? 1.5 : 2),
        boxShadow: isDarkMode ? 'none' : 'inset 0 -1px 0 rgba(0,0,0,0.08)',
        borderBottom: isDarkMode ? '1px solid rgba(255,255,255,0.05)' : `1px solid ${alpha(theme.palette.divider, 0.08)}`,
        borderRadius: theme.spacing(0),
        position: 'relative',
        flexShrink: 0, // Prevent header from shrinking vertically
        overflow: 'hidden', // Ensure the pseudo-element doesn't cause overflow
        '&::after': {
          content: '""',
          position: 'absolute',
          left: 0,
          right: 0,
          ...(isDarkMode ? {
            top: 0,
            height: '2px',
            background: `linear-gradient(90deg, ${alpha(graphColors.main, 0.3)}, ${alpha(graphColors.main, 0)})`
          } : {
            bottom: 0,
            height: '3px',
            background: `linear-gradient(90deg, ${alpha(graphColors.main, 0.1)}, ${alpha(graphColors.main, 0)})`
          })
        }
      }}>
        {/* Lemma and Audio Button */}
        <Stack direction="row" spacing={1} alignItems="flex-start" flexWrap="nowrap" sx={{ mb: theme.spacing(0) }}>
          {/* Lemma Typography - Fully responsive */}
          <Typography 
            variant="h3" 
            component="h1" 
            className="word-details-header-lemma" 
            sx={{ 
              // Core flex properties
              flexGrow: 1, 
              minWidth: 0, 
              // Text wrapping and sizing
              overflowWrap: 'break-word',
              wordBreak: 'break-word', 
              // Responsive font sizing - REDUCED FOR MOBILE
              fontSize: {
                xs: '1.3rem',   // Mobile (was 1.5rem)
                sm: '1.5rem',  // Tablet (was 1.75rem)
                md: '2rem',     
                lg: '2.125rem', 
              },
              // Additional styling
              fontWeight: 700, 
              letterSpacing: '-0.01em',
              color: isDarkMode ? '#ffffff' : theme.palette.text.primary,
              position: 'relative', // For pseudo-element
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: { xs: -1, sm: -2 }, // Adjusted bottom offset
                left: 0,
                width: '30px', // Shorter underline
                height: '2px',
                bgcolor: mainColor,
                display: 'block',
                borderRadius: '2px',
              }
            }}
          >
            {wordData.lemma}
          </Typography>
          {/* Audio Button - Prevent shrinking */}
          {hasAudio && (
            <IconButton
              onClick={playAudio}
              size="medium"
              title={isAudioPlaying ? "Stop Audio" : "Play Audio"}
              sx={{ 
                flexShrink: 0, // Prevent button from shrinking
                color: mainColor, 
                mt: 0.5, 
                bgcolor: 'transparent',
                '&:hover': { 
                  bgcolor: alpha(mainColor, 0.1) 
                } 
              }}
            >
              {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
            </IconButton>
          )}
        </Stack>

        {/* Pronunciations Section */}
        <Stack spacing={1.5} sx={{ mb: theme.spacing(0) }}> { /* Keep mb at 0 */ }
          {/* IPA Pronunciation Display (uses the case-insensitive check above) */}
          {ipaPronunciation && (
            <Typography 
              variant="body1" // Reduced from h6
              sx={{ 
                fontStyle: 'italic', 
                pl: theme.spacing(0.5),
                fontSize: isMobile ? '0.9rem' : '1rem', // Smaller font on mobile
                color: isDarkMode ? alpha(headerTextColor, 0.85) : alpha(theme.palette.text.primary, 0.75),
                display: 'inline-block',
                py: 0.5,
                px: 1,
                borderRadius: '4px',
                bgcolor: 'transparent',
                border: 'none',
              }}
            >
              /{ipaPronunciation.value}/
            </Typography>
          )}

          {/* Other Pronunciation Types (make filter case-insensitive) */}
          {wordData.pronunciations && wordData.pronunciations.filter(p => 
            p.type?.toLowerCase() !== 'ipa' && 
            p.type?.toLowerCase() !== 'audio' &&
            p.type?.toLowerCase() !== 'rhyme' // Exclude rhymes
          ).length > 0 && (
            <Box sx={{ mt: 1 }}>
              <Stack direction="row" spacing={2} flexWrap="wrap">
                {wordData.pronunciations.filter(p => 
                  p.type?.toLowerCase() !== 'ipa' && 
                  p.type?.toLowerCase() !== 'audio' &&
                  p.type?.toLowerCase() !== 'rhyme' // Exclude rhymes again for mapping
                ).map((pron, index) => (
                  <Box key={index}>
                    <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.7), fontWeight: 500 }}>
                      {/* Capitalize type name for display */}
                      {pron.type ? pron.type.charAt(0).toUpperCase() + pron.type.slice(1) : 'Pronunciation'}
                    </Typography>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {pron.value}
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          )}
        </Stack>

        {/* Baybayin - Show only if available */}
        {wordData.has_baybayin && wordData.baybayin_form && wordData.baybayin_form.trim() !== '' && (
          <Box sx={{ my: 0, mt: theme.spacing(isMobile ? 1 : 1.5) }}> 
            <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.75), display: 'block', mb: 0.25, fontSize: isMobile ? '0.7rem' : '0.75rem' }}>
              Baybayin Script
            </Typography>
            <div 
              className="baybayin-text"
              style={{
                fontFamily: "'Noto Sans Tagalog', 'Arial Unicode MS', 'Noto Sans', sans-serif !important", // Corrected font name
                fontSize: isMobile ? '1.5rem' : '2rem', // Reduced size on mobile
                padding: isMobile ? '4px 8px' : '8px 12px',
                background: isDarkMode ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.5)',
                borderRadius: '4px',
                display: 'inline-block',
                lineHeight: 1.2,
                minHeight: isMobile ? '36px' : '48px',
                marginTop: '4px',
                color: 'inherit',
                border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.05)' : '1px solid rgba(0, 0, 0, 0.05)'
              }}
            >
              {wordData.baybayin_form}
            </div>
          </Box> // Ensure this Box is closed
        )}

        {/* Badlit Form - Show only if available */}
        {wordData.badlit_form && wordData.badlit_form.trim() !== '' && (
          <Box sx={{ my: 0, mt: theme.spacing(isMobile ? 1 : 1.5) }}> 
            <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.75), display: 'block', mb: 0.25, fontSize: isMobile ? '0.7rem' : '0.75rem' }}>
              Badlit Form
            </Typography>
            <div 
              className="badlit-text"
              style={{
                fontSize: isMobile ? '1.2rem' : '1.5rem', // Reduced size on mobile
                padding: isMobile ? '4px 8px' : '8px 12px',
                background: isDarkMode ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.5)',
                borderRadius: '4px',
                display: 'inline-block',
                lineHeight: 1.2,
                minHeight: isMobile ? '36px' : '48px',
                marginTop: '4px',
                color: 'inherit',
                border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.05)' : '1px solid rgba(0, 0, 0, 0.05)'
              }}
            >
              {wordData.badlit_form}
            </div>
          </Box> // Ensure this Box is closed
        )}

        {/* Additional info - romanized form, language - Enhanced styling */}
        <Stack 
          direction="row" 
          spacing={isMobile ? 1.5 : 3} // Reduced spacing on mobile
          sx={{ 
            mt: 1, 
            pt: 1, 
            borderTop: `1px solid ${isDarkMode ? alpha('#fff', 0.05) : alpha('#000', 0.03)}`,
          }}
        >
          {wordData.language_code && (
            <Box> 
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(mainColor, 0.8),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.25, // Reduced margin
                  fontSize: isMobile ? '0.7rem' : '0.75rem'
                }}
              >
                Language
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 600,
                  display: 'inline-block',
                  px: 0.75, // Reduced padding
                  py: 0.1,  // Reduced padding
                  borderRadius: '3px',
                  bgcolor: isDarkMode ? alpha(mainColor, 0.15) : alpha(mainColor, 0.08),
                  color: isDarkMode ? alpha(headerTextColor, 0.9) : mainColor,
                  border: `1px solid ${isDarkMode ? alpha(mainColor, 0.2) : alpha(mainColor, 0.15)}`,
                  fontSize: isMobile ? '0.75rem' : '0.875rem' // Reduced font size
                }}
              >
                {wordData.language_code.toUpperCase()}
              </Typography>
            </Box> 
          )}
          
          {wordData.romanized_form && (
            <Box> // REMOVED COMMENT
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(theme.palette.text.primary, 0.7),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.25, // Reduced margin
                  fontSize: isMobile ? '0.7rem' : '0.75rem'
                }}
              >
                Romanized Form
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 500,
                  color: isDarkMode ? alpha(headerTextColor, 0.95) : theme.palette.text.primary,
                  fontSize: isMobile ? '0.8rem' : '0.875rem' // Reduced font size
                }}
              >
                {wordData.romanized_form}
              </Typography>
            </Box>
          )}
          
          {wordData.created_at && (
            <Box> // REMOVED COMMENT
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(theme.palette.text.primary, 0.7),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.25, // Reduced margin
                  fontSize: isMobile ? '0.7rem' : '0.75rem'
                }}
              >
                Added
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 500,
                  color: isDarkMode ? alpha(headerTextColor, 0.95) : theme.palette.text.primary,
                  fontSize: isMobile ? '0.8rem' : '0.875rem' // Reduced font size
                }}
              >
                {new Date(wordData.created_at).toLocaleDateString()}
              </Typography>
            </Box>
          )}
        </Stack>

        {/* Tags with enhanced styling */}
        {tags.length > 0 && (
          <Stack 
            direction="row" 
            spacing={1} 
            useFlexGap 
            flexWrap="wrap" 
            sx={{ mt: theme.spacing(2) }}
          >
            {tags.map((tag) => (
              <Chip
                key={tag}
                label={tag}
                size="small"
                sx={{
                  color: isDarkMode ? alpha(headerTextColor, 0.9) : theme.palette.text.primary,
                  borderColor: isDarkMode ? alpha(headerTextColor, 0.3) : alpha(theme.palette.text.primary, 0.2),
                  bgcolor: isDarkMode ? alpha(headerTextColor, 0.07) : alpha(theme.palette.background.paper, 0.5),
                  '& .MuiChip-label': { fontWeight: 500 },
                  height: 'auto',
                  padding: theme.spacing(0.25, 0.5)
                }}
                variant="outlined"
              />
            ))}
          </Stack>
        )}
      </Box>
    );
  };

  const renderDefinitionsTab = () => {
    if (!wordData.definitions || wordData.definitions.length === 0) {
      return <Alert severity="info">No definitions available for this word.</Alert>;
    }

    const definitionsByPosThenSource: { [pos: string]: { [source: string]: Definition[] } } = {};
    wordData.definitions.forEach((def: Definition) => {
      const posKey = def.part_of_speech?.name_en || def.part_of_speech?.code || 'Other';
      const sourceKey = (def.sources && def.sources.length > 0) ? def.sources[0] : 'Unknown Source'; 
      if (!definitionsByPosThenSource[posKey]) {
        definitionsByPosThenSource[posKey] = {};
      }
      if (!definitionsByPosThenSource[posKey][sourceKey]) {
        definitionsByPosThenSource[posKey][sourceKey] = [];
      }
      definitionsByPosThenSource[posKey][sourceKey].push(def);
    });

    return (
      // Use pt instead of wrapping Box for top padding
      <Box sx={{ pt: isMobile ? 0.5 : 1, width: '100%', maxWidth: '100%' }}> 
        {Object.entries(definitionsByPosThenSource).map(([posName, defsBySource]) => {
          return (
          <Box key={posName} sx={{ mb: isMobile ? 2 : 3, width: '100%', maxWidth: '100%' }}>
            {/* Part of Speech Header - English/Code Only */}
            <Typography 
              variant="subtitle1" 
              component="h3" 
              sx={{ 
                color: graphColors.main, 
                fontWeight: 600,
                pb: isMobile ? 0.5 : 1,
                fontSize: isMobile ? '0.9rem' : '1rem', // Reduced font size
                borderBottom: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
                mb: isMobile ? 1 : 1.5,
                width: '100%',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
            >
              {posName} 
            </Typography>
            
            {/* Iterate through definitions grouped by source */}
            {Object.entries(defsBySource).map(([sourceName, defs]) => {
              // Filtering logic
              const seenBaseDefinitions = new Set<string>();
              const filteredDefs = defs.filter((def) => {
                const baseText = stripLeadingNumber(def.text);
                if (seenBaseDefinitions.has(baseText)) {
                  return false; 
                }
                seenBaseDefinitions.add(baseText);
                return true; 
              });
              
              if (filteredDefs.length === 0) return null;

              return (
              <Box key={sourceName} sx={{ mb: 2, width: '100%', maxWidth: '100%' }}>
                <List disablePadding>
                  {filteredDefs.map((def: Definition, index: number) => {
                    const isLastDefinitionForSource = index === filteredDefs.length - 1;
                    const tagalogPos = def.part_of_speech?.name_tl;
                    const showTagalogPos = tagalogPos && tagalogPos !== posName;

                    return (
                    <ListItem 
                      key={def.id || index} 
                      alignItems="flex-start" 
                      sx={{ 
                        flexDirection: 'column', 
                        gap: 0.5, 
                        py: isMobile ? 1 : 1.5, 
                        pl: 0, 
                        pr: 0, // Ensure no right padding
                        position: 'relative', 
                        pb: isLastDefinitionForSource && sourceName !== 'Unknown Source' ? (isMobile ? 2.5 : 3) : (isMobile ? 1 : 1.5), 
                      }}
                    >
                      {/* Definition text */}
                      <ListItemText
                        primaryTypographyProps={{ 
                          variant: 'body1', 
                          fontWeight: 500, 
                          fontSize: isMobile ? '0.85rem' : '1rem', // Reduced font size
                          pl: 0.5, 
                          lineHeight: isMobile ? 1.5 : 1.6 // Increased lineHeight
                        }}
                        primary={def.text}
                      />

                      {/* Display Tagalog POS if available and different */}
                      {showTagalogPos && (
                        <Typography 
                          variant="caption" 
                          sx={{ 
                            pl: 0.5, // Indent slightly
                            mt: 0.5, // Add space below definition text
                            fontSize: isMobile ? '0.7rem' : '0.75rem', // Reduced font size
                            color: 'text.secondary', 
                            fontStyle: 'italic' 
                          }}
                        >
                          (TL: {tagalogPos})
                        </Typography>
                      )}
                      
                      {/* Examples */}
                      {def.examples && def.examples.length > 0 && (
                        <Box sx={{ pl: 2, mb: 1 }}>
                          {def.examples.map((example, exIndex) => (
                            <Typography 
                              key={exIndex} 
                              variant="body2" 
                              sx={{ 
                                fontStyle: 'italic', 
                                color: 'text.secondary',
                                fontSize: isMobile ? '0.8rem' : '0.875rem', // Reduced font size
                                mb: exIndex < def.examples.length - 1 ? 0.5 : 0,
                                position: 'relative',
                                pl: 3,
                                lineHeight: isMobile ? 1.4 : 1.5, // Increased lineHeight
                                '&:before': {
                                  content: '"""',
                                  position: 'absolute',
                                  left: 0,
                                  color: alpha(theme.palette.text.primary, 0.4),
                                  fontSize: '1.5rem',
                                  lineHeight: 1,
                                  top: -5,
                                }
                              }}
                            >
                              {example}
                            </Typography>
                          ))}
                        </Box>
                      )}
                      
                      {/* Usage notes */}
                      {def.usage_notes && def.usage_notes.length > 0 && (
                        <Box sx={{ pl: 1, mb: 1 }}>
                          <Typography 
                            variant="caption" 
                            component="div" 
                            sx={{ fontWeight: 500, mb: 0.5, fontSize: isMobile ? '0.75rem' : '0.8rem' }}
                          >
                            Usage Notes:
                          </Typography>
                          {def.usage_notes.map((note, noteIndex) => (
                            <Typography 
                              key={noteIndex} 
                              variant="body2" 
                              sx={{ color: 'text.secondary', fontSize: isMobile ? '0.8rem' : '0.875rem' }}
                            >
                              {note}
                            </Typography>
                          ))}
                        </Box>
                      )}
                      
                      {/* Definition tags */}
                      {def.tags && def.tags.length > 0 && (
                        <Stack 
                          direction="row" 
                          spacing={0.5} 
                          useFlexGap 
                          flexWrap="wrap" 
                          sx={{ mt: 0.5 }}
                        >
                          {def.tags.map((tag, tagIndex) => (
                            <Chip
                              key={tagIndex}
                              label={tag}
                              size="small"
                              sx={{
                                fontSize: isMobile ? '0.65rem' : '0.7rem', // Reduced font size
                                height: 'auto',
                                padding: theme.spacing(0.25, 0),
                                bgcolor: alpha(theme.palette.primary.main, 0.1),
                                color: theme.palette.primary.main,
                                '& .MuiChip-label': { 
                                  px: 0.75, 
                                  py: 0.2 
                                }
                              }}
                            />
                          ))}
                        </Stack>
                      )}

                      {/* Source Chip */}
                      {isLastDefinitionForSource && sourceName !== 'Unknown Source' && (
                        <Chip
                          label={`${sourceName}`}
                          size="small"
                          variant="outlined"
                          sx={{
                            position: 'absolute', // Position relative to ListItem
                            bottom: theme.spacing(isMobile ? 0.25 : 0.5),
                            right: theme.spacing(isMobile ? 0.25 : 0.5),
                            fontSize: isMobile ? '0.65rem' : '0.7rem', // Reduced font size
                            height: 'auto',
                            padding: theme.spacing(0.25, 0),
                            borderColor: alpha(graphColors.related, 0.4),
                            color: alpha(graphColors.related, 0.9),
                            bgcolor: alpha(graphColors.related, 0.04),
                            maxWidth: 'calc(100% - 16px)', // Prevent overflow
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            zIndex: 1, // Ensure it's above other content if overlap occurs
                            '& .MuiChip-label': { 
                              px: 1, 
                              py: 0.25 
                            }
                          }}
                        />
                      )}
                    </ListItem>
                    );
                  })}
                </List>
              </Box>
            );
          })}
          </Box>
        );
      })}
      </Box>
    );
  };

  const renderRelationsTab = () => {
    // Check for defined relations or default to empty arrays
    const incoming_relations = wordData.incoming_relations || [];
    const outgoing_relations = wordData.outgoing_relations || [];
    
    // Get semantic network data safely
    const semanticNetworkNodes = wordData.semantic_network?.nodes || [];
    const semanticNetworkLinks = wordData.semantic_network?.links || [];
    
    // Enhanced fallback logic: Use semantic network if regular relations are empty OR there's an error message
    // This handles the case where the backend returns an error due to the missing relation_data column
    const hasError = typeof wordData.server_error === 'string' && wordData.server_error.includes('database error');
    const useSemanticNetwork = 
      ((incoming_relations.length === 0 && outgoing_relations.length === 0) || hasError) && 
      semanticNetworkLinks.length > 0;
    
    // Log the fallback status
    if (hasError) {
      console.log("Using semantic network fallback due to server error:", wordData.server_error);
    }
      
    // Helper function to create relation objects from semantic network data
    function createRelationsFromNetwork() {
      console.log("Creating relations from semantic network as fallback");
      const mainWord = wordData.lemma;
      const relations: any[] = [];
      
      semanticNetworkLinks.forEach(link => {
        // Find the connected node
        const targetNode = semanticNetworkNodes.find(
          n => n.id === (typeof link.target === 'object' ? link.target.id : link.target)
        );
        
        if (!targetNode) return;
        
        // Calculate degree/distance (default to 1 for direct connections)
        const degree = link.distance || link.degree || 1;
        
        // Create a relation object
        relations.push({
          id: `semantic-${targetNode.id}`,
          relation_type: link.type || 'related',
          degree: degree,
          wordObj: {
            id: targetNode.id,
            lemma: targetNode.label || targetNode.word || String(targetNode.id),
            has_baybayin: targetNode.has_baybayin || false,
            baybayin_form: targetNode.baybayin_form || null
          }
        });
      });
      
      console.log(`Created ${relations.length} fallback relations from semantic network`);
      return relations;
    }
    
    // Create relations from semantic network if needed
    const semanticRelations = useSemanticNetwork ? createRelationsFromNetwork() : [];
    
    // Check if there are any standard relations or fallback relations
    const hasStandardRelations = (incoming_relations.length > 0 || outgoing_relations.length > 0) && !hasError;
    const hasAnyRelations = hasStandardRelations || semanticRelations.length > 0 || (wordData.root_affixations && wordData.root_affixations.length > 0) || (wordData.affixed_affixations && wordData.affixed_affixations.length > 0);
      
    if (!hasAnyRelations) {
        return <Alert severity="info">No relations or affixations available for this word.</Alert>;
    }

    // Helper to render Affixations Section
    const renderAffixations = () => {
      const rootAffix = wordData.root_affixations || [];
      const affixedAffix = wordData.affixed_affixations || [];

      if (rootAffix.length === 0 && affixedAffix.length === 0) return null;

      return (
        <StyledAccordion sx={{ mb: 2 }}> {/* Add margin bottom */}
          <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Affixations</Typography>
          </StyledAccordionSummary>
          <StyledAccordionDetails>
            {rootAffix.length > 0 && (
              <Box sx={{ mb: affixedAffix.length > 0 ? 2 : 0 }}> {/* Add margin if both sections exist */}
                <Typography variant="caption" sx={{ fontWeight: 500 }}>Derived via Affixation:</Typography>
                <List dense disablePadding sx={{ pl: 1 }}>
                  {rootAffix.map((affix: Affixation, index: number) => (
                    <ListItem key={affix.id || index} disableGutters sx={{ py: 0.25 }}>
                      <ListItemText
                        primary={affix.affixed_word?.lemma || 'Unknown'}
                        secondary={`(as ${affix.affix_type})`}
                        primaryTypographyProps={{ component: 'span', variant: 'body2', sx: { cursor: 'pointer', textDecoration: 'underline', color: theme.palette.primary.main, mr: 1 }, onClick: () => affix.affixed_word && onWordClick(affix.affixed_word) }}
                        secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                        sx={{ m: 0 }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
            {affixedAffix.length > 0 && (
              <Box>
                <Typography variant="caption" sx={{ fontWeight: 500 }}>Derived from Root:</Typography>
                <List dense disablePadding sx={{ pl: 1 }}>
                  {affixedAffix.map((affix: Affixation, index: number) => (
                    <ListItem key={affix.id || index} disableGutters sx={{ py: 0.25 }}>
                      <ListItemText
                        primary={affix.root_word?.lemma || 'Unknown'}
                        secondary={`(using ${affix.affix_type})`}
                        primaryTypographyProps={{ component: 'span', variant: 'body2', sx: { cursor: 'pointer', textDecoration: 'underline', color: theme.palette.primary.main, mr: 1 }, onClick: () => affix.root_word && onWordClick(affix.root_word) }}
                        secondaryTypographyProps={{ variant: 'caption', color: 'text.secondary' }}
                        sx={{ m: 0 }}
                      />
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </StyledAccordionDetails>
        </StyledAccordion>
      );
    };

    // Group all relations by type - use semantic relations if in fallback mode
    const allRelations = useSemanticNetwork ? semanticRelations : [...incoming_relations, ...outgoing_relations];
    
    // Add degree information for regular relations
    const relationsWithDegree = allRelations.map(rel => ({
      ...rel, 
      wordObj: rel.source_word || rel.target_word || rel.wordObj, // Ensure wordObj exists
      degree: rel.degree || 1 // Default to 1 (direct connection) if not specified
    }));
    
    // Filter out self-references (where the related word is the same as the main word)
    const filteredRelations = relationsWithDegree.filter(rel => {
      return rel.wordObj?.lemma !== wordData.lemma;
    });
    
    // Define relation categories and their types
    const relationCategories = [
      {
        name: "Origin",
        types: ["root", "etymology", "cognate", "root_of", "isahod", "derived", "derived_from", "sahod", "affix", "derivative"],
        color: relationColors.root
      },
      {
        name: "Meaning",
        types: ["synonym", "antonym", "related", "similar", "kaugnay", "kahulugan", "kasalungat"],
        color: relationColors.synonym
      },
      {
        name: "Form",
        types: ["variant", "spelling", "abbreviation", "form_of", "regional_variant", "itapat", "atapat", "inatapat"],
        color: relationColors.variant
      },
      {
        name: "Hierarchy",
        types: ["hypernym", "hyponym", "meronym", "holonym", "taxonomic", "part_whole", "component", "component_of"],
        color: relationColors.hypernym
      },
      {
        name: "Usage",
        types: ["usage"],
        color: relationColors.usage
      },
      {
        name: "Other",
        types: ["other", "associated"],
        color: relationColors.other
      }
    ];
    
    // COMPLETE RESTRUCTURE: Deduplicate first, then categorize once
    
    // Step 1: Group by word lemma, collecting all relation types for each word
    const wordMap = new Map<string, any>();
    
    filteredRelations.forEach(relation => {
      const wordLemma = relation.wordObj?.lemma;
      if (!wordLemma) return;
      
      if (!wordMap.has(wordLemma)) {
        // First time seeing this word, create a new entry
        const newRelation = { 
          ...relation,
          relationTypes: [relation.relation_type],
          originalRelation: relation
        };
        wordMap.set(wordLemma, newRelation);
      } else {
        // Word exists, just add the relation type if new
        const existingRelation = wordMap.get(wordLemma);
        if (!existingRelation.relationTypes.includes(relation.relation_type)) {
          existingRelation.relationTypes.push(relation.relation_type);
        }
        
        // If this relation has a smaller degree, prefer it
        if (relation.degree < existingRelation.degree) {
          existingRelation.degree = relation.degree;
        }
      }
    });
    
    // Step 2: Determine primary category for each word 
    // (based on priority of relation types)
    const priorityOrder: string[] = [];
    relationCategories.forEach(category => {
      category.types.forEach(type => {
        if (!priorityOrder.includes(type)) {
          priorityOrder.push(type);
        }
      });
    });
    
    // Step 3: Place each word in exactly ONE category based on its highest priority relation type
    const categorizedWords: Record<string, any[]> = {};
    
    // Initialize categories
    relationCategories.forEach(category => {
      categorizedWords[category.name] = [];
    });
    
    // Debug the data we're working with
    console.log("Deduplicating relations. Unique words count:", wordMap.size);
    console.log("Word lemmas:", Array.from(wordMap.keys()));
    
    // Process each unique word
    wordMap.forEach((relation) => {
      // Sort relation types by priority
      const sortedTypes = [...relation.relationTypes].sort((a, b) => {
        const aIndex = priorityOrder.indexOf(a);
        const bIndex = priorityOrder.indexOf(b);
        
        if (aIndex >= 0 && bIndex >= 0) return aIndex - bIndex;
        if (aIndex >= 0) return -1;
        if (bIndex >= 0) return 1;
        return 0;
      });
      
      // Use highest priority relation type to determine category
      const primaryType = sortedTypes[0];
      let categoryPlaced = false;
      
      // Find which category this primary type belongs to
      for (const category of relationCategories) {
        if (category.types.includes(primaryType)) {
          relation.primaryCategory = category.name;
          relation.primaryType = primaryType;
          categorizedWords[category.name].push(relation);
          categoryPlaced = true;
          break;
        }
      }
      
      // If not placed in any specific category, put in Other
      if (!categoryPlaced) {
        relation.primaryCategory = "Other";
        relation.primaryType = primaryType || "other";
        categorizedWords["Other"].push(relation);
      }
    });
    
    // Final deduplication check - ensure words appear only once across all categories
    const seenWords = new Set<string>();
    Object.keys(categorizedWords).forEach(category => {
      categorizedWords[category] = categorizedWords[category].filter(relation => {
        const lemma = relation.wordObj?.lemma;
        if (!lemma || seenWords.has(lemma)) {
          return false; // Skip if we've seen this word before
        }
        seenWords.add(lemma);
        return true;
      });
    });
    
    // Calculate the number of unique related words (already deduplicated)
    const uniqueRelatedWords = wordMap.size;
    
    // Filter out empty categories
    const nonEmptyCategories = relationCategories.filter(
      category => categorizedWords[category.name].length > 0
    );
                
                return (
      <Box sx={{ pt: theme.spacing(1), width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
        {/* ADDED Call to render Affixations */} 
        {renderAffixations()}

        {/* Show error banner if using fallback due to server error */}
        {hasError && (
          <Alert severity="warning" sx={{ mb: theme.spacing(2), width: '100%' }}>
            {wordData.server_error || 'Server database error. Using semantic network relationships as fallback.'}
          </Alert>
        )}
        
        {/* Elegant relationship summary */}
        <Box 
          sx={{ 
            mb: theme.spacing(3),
            pb: theme.spacing(1),
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 1,
            width: '100%'
          }}
        >
          <Typography 
            variant="subtitle1" 
            sx={{ 
              fontWeight: 600, 
              color: graphColors.main,
            }}
          >
            {uniqueRelatedWords} related {uniqueRelatedWords === 1 ? 'word' : 'words'}
          </Typography>
          
          {useSemanticNetwork && (
                  <Chip
              label="Using semantic network data"
              size="small"
              variant="outlined"
              color="info"
              sx={{ height: 24 }}
            />
          )}
        </Box>
        
        {/* Relation categories */}
        <Box sx={{ mb: theme.spacing(3), width: '100%', maxWidth: '100%' }}>
          {nonEmptyCategories.map((category, categoryIndex) => {
            const relations = categorizedWords[category.name];
            
            return (
              <Box 
                key={category.name} 
                sx={{ 
                  mb: theme.spacing(2),
                  pb: theme.spacing(2),
                  borderBottom: categoryIndex < nonEmptyCategories.length - 1 ? 
                    `1px solid ${alpha(theme.palette.divider, 0.1)}` : 'none',
                  width: '100%',
                  maxWidth: '100%'
                }}
              >
                <Typography 
                  variant="subtitle2" 
                  sx={{ 
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    mb: theme.spacing(1.5),
                    pb: theme.spacing(0.5),
                    borderBottom: `1px solid ${alpha(category.color, 0.2)}`,
                    color: theme.palette.text.primary,
                    fontWeight: 600
                  }}
                >
                  <Box 
                    component="span"
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: category.color,
                      boxShadow: `0 0 0 2px ${alpha(category.color, 0.2)}`
                    }}
                  />
                  {category.name}
                  <Typography 
                    component="span" 
                    sx={{ 
                      color: alpha(theme.palette.text.secondary, 0.6),
                      fontSize: '0.8rem',
                      fontWeight: 'normal'
                    }}
                  >
                    ({relations.length})
                  </Typography>
                </Typography>
                
                {/* Group relations by their specific types within the category */}
                <Box sx={{ pl: theme.spacing(2) }}>
                  {(() => {
                    // Group relations by their primary type
                    const typeGroups: Record<string, any[]> = {};
                    const typeSeenWords = new Set<string>();
                    
                    relations.forEach(relation => {
                      const type = relation.primaryType || 'other';
                      const lemma = relation.wordObj?.lemma;
                      
                      // Skip if we've seen this word within this category
                      if (!lemma || typeSeenWords.has(lemma)) return;
                      typeSeenWords.add(lemma);
                      
                      if (!typeGroups[type]) {
                        typeGroups[type] = [];
                      }
                      typeGroups[type].push(relation);
                    });
                    
                    // Sort types alphabetically
                    const sortedTypes = Object.keys(typeGroups).sort();
                    
                    return sortedTypes.map(relationType => {
                      const typeRelations = typeGroups[relationType];
                      const relColor = relationColors[relationType] || category.color;
                      
      return (
                        <Box key={relationType} sx={{ mb: theme.spacing(1.5) }}>
                          <Typography 
                            variant="body2" 
                            sx={{ 
                              fontWeight: 500,
                              fontSize: '0.85rem',
                              color: alpha(relColor, 0.9),
                              mb: theme.spacing(0.75),
                              display: 'flex',
                              alignItems: 'center',
                              gap: 0.75,
                              py: 0.5
                            }}
                          >
                            <Box 
                              component="span"
                              sx={{
                                width: 8,
                                height: 8,
                                borderRadius: '50%',
                                bgcolor: relColor,
                                boxShadow: `0 0 0 2px ${alpha(relColor, 0.2)}`
                              }}
                            />
                            {formatRelationType(relationType)}
                            <Typography 
                              component="span" 
                              sx={{ 
                                color: alpha(relColor, 0.6),
                                fontSize: '0.75rem',
                                fontWeight: 'normal'
                              }}
                            >
                              ({typeRelations.length})
                            </Typography>
                          </Typography>
                          
                          <Stack 
                            direction="row" 
                            spacing={1} // Keep spacing for consistency between items
                            useFlexGap // Allow wrapping
                            flexWrap="wrap"
                            sx={{ 
                              mb: theme.spacing(1.5), 
                              pl: theme.spacing(1), 
                              width: '100%', 
                              maxWidth: '100%',
                              gap: isMobile ? 1 : 0.75 // Add/adjust gap for spacing between wrapped lines (more on mobile)
                            }}
                          >
                            {typeRelations.map((relation, index) => {
                              const wordObj = relation.wordObj;
                              if (!wordObj) return null;
                              
                              // Check if this word has multiple relation types
                              const hasMultipleTypes = relation.relationTypes && relation.relationTypes.length > 1;
              
              return (
                <Chip
                                  key={`${wordObj.id || index}`}
                                  label={
                                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                      {/* Degree indicator dot */}
                                      {relation.degree > 1 && (
                                        <Box 
                                          component="span"
                                          sx={{ 
                                            width: 6, 
                                            height: 6, 
                                            borderRadius: '50%', 
                                            backgroundColor: 'rgba(0,0,0,0.4)',
                                            display: 'inline-block',
                                            mr: 0.8,
                                            boxShadow: theme.palette.mode === 'dark' ? '0 0 0 1.5px rgba(255,255,255,0.4)' : '0 0 0 1.5px rgba(0,0,0,0.2)',
                                            flexShrink: 0
                                          }}
                                          title={`Indirect connection (${relation.degree} degrees of separation)`}
                                        />
                                      )}
                                      <span style={{ 
                                        overflow: 'hidden', 
                                        textOverflow: 'ellipsis', 
                                        whiteSpace: 'nowrap', 
                                        maxWidth: '130px'
                                      }}>
                                        {wordObj.lemma || wordObj.word}
                                      </span>
                                      {hasMultipleTypes && (
                                        <Typography 
                                          component="span" 
                                          sx={{ 
                                            fontSize: '0.7rem', 
                                            ml: 0.5,
                                            opacity: 0.7,
                                            fontWeight: 400,
                                            flexShrink: 0
                                          }}
                                        >
                                          ({relation.relationTypes.length})
                                        </Typography>
                                      )}
                                    </Box>
                                  }
                                  onClick={() => onWordClick(wordObj.lemma || wordObj.word)}
                                  variant="outlined"
                                  sx={{
                                    // Base styles (Light mode defaults)
                                    borderColor: alpha(relColor, 0.5),
                                    color: relColor,
                                    fontSize: '0.75rem',
                                    height: 'auto',
                                    padding: theme.spacing(0.25, 0),
                                    bgcolor: alpha(relColor, 0.05),
                                    my: 0.5,
                                    maxWidth: '100%',
                                    '& .MuiChip-label': { 
                                      px: 1, 
                                      py: 0.25, 
                                      fontWeight: 500,
                                      width: '100%',
                                      maxWidth: '100%',
                                      overflow: 'hidden'
                                    },
                                    // Dark Mode Overrides
                                    ...(theme.palette.mode === 'dark' && {
                                      bgcolor: alpha(relColor, 0.35), // More visible base color
                                      color: theme.palette.common.white, // Use white for contrast
                                      borderColor: alpha(relColor, 0.7), // More visible border
                                    }),
                                    // Hover styles
                                    '&:hover': {
                                      // Light mode hover
                                      backgroundColor: alpha(relColor, 0.1),
                                      borderColor: relColor,
                                      // Dark Mode hover overrides
                                      ...(theme.palette.mode === 'dark' && {
                                        backgroundColor: alpha(relColor, 0.6), // Intensify background
                                        borderColor: alpha(relColor, 0.9), // Intensify border
                                      }),
                                    }
                                  }}
                />
              );
            })}
                          </Stack>
                        </Box>
                      );
                    });
                  })()}
                </Box>
              </Box>
            );
          })}
        </Box>
        
        {/* Display completeness info if available */}
        {wordData.data_completeness && (
          <Box sx={{ mt: theme.spacing(3), pt: theme.spacing(2), borderTop: `1px solid ${theme.palette.divider}` }}>
            <Typography variant="subtitle2" sx={{ mb: theme.spacing(1) }}>
              Data Completeness
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 1 }}>
              {Object.entries(wordData.data_completeness).map(([key, value]) => (
                <Chip
                  key={key}
                  label={key.replace(/_/g, ' ')}
                  color={value ? "success" : "default"}
                  variant={value ? "filled" : "outlined"}
                  size="small"
                  sx={{ justifyContent: 'flex-start' }}
                />
              ))}
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  const renderFormsAndTemplatesTab = () => {
    const forms = wordData.forms || [];
    const templates = wordData.templates || [];
    const rhymes = wordData.pronunciations?.filter(p => p.type?.toLowerCase() === 'rhyme') || [];

    if (forms.length === 0 && templates.length === 0 && rhymes.length === 0) {
      return <Alert severity="info">No forms, templates, or rhyme information available.</Alert>;
    }

    return (
      // Adjust top padding for mobile
      <Box sx={{ pt: isMobile ? theme.spacing(0.5) : theme.spacing(1) }}>
        {/* Forms Section */}
        {forms.length > 0 && (
          <StyledAccordion defaultExpanded sx={{ mb: 2 }}>
            <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" sx={{ fontWeight: 500, fontSize: isMobile ? '0.85rem' : '0.875rem' }}>Word Forms</Typography>
            </StyledAccordionSummary>
            <StyledAccordionDetails sx={{ px: isMobile ? 1 : 2 }}>
              <List dense={isMobile} disablePadding>
                {forms.map((form: WordForm, index: number) => (
                  <ListItem key={form.id || index} disableGutters sx={{ py: isMobile ? 0.1 : 0.25 }}>
                    <ListItemText 
                      primary={form.form} 
                      primaryTypographyProps={{ sx: { fontSize: isMobile ? '0.8rem' : '0.875rem' }}}
                    />
                    <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                       {form.is_canonical && <Chip label="Canonical" size="small" color="primary" variant="outlined" sx={{ height: 'auto', fontSize: isMobile ? '0.6rem' : '0.65rem', py: 0.1, px: 0.25 }} />}
                       {form.is_primary && <Chip label="Primary" size="small" color="secondary" variant="outlined" sx={{ height: 'auto', fontSize: isMobile ? '0.6rem' : '0.65rem', py: 0.1, px: 0.25 }} />}
                       {form.tags && Object.entries(form.tags).map(([key, value]) => (
                          <Chip key={key} label={`${key}: ${value}`} size="small" sx={{ height: 'auto', fontSize: isMobile ? '0.6rem' : '0.65rem', py: 0.1, px: 0.25 }} />
                       ))}
                    </Stack>
                  </ListItem>
                ))}
              </List>
            </StyledAccordionDetails>
          </StyledAccordion>
        )}

        {/* Templates Section */}
        {templates.length > 0 && (
          <StyledAccordion defaultExpanded>
            <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" sx={{ fontWeight: 500, fontSize: isMobile ? '0.85rem' : '0.875rem' }}>Word Templates</Typography>
            </StyledAccordionSummary>
            <StyledAccordionDetails sx={{ px: isMobile ? 1 : 2 }}>
              <List dense={isMobile} disablePadding>
                {templates.map((template: WordTemplate, index: number) => (
                  <ListItem key={template.id || index} disableGutters sx={{ py: isMobile ? 0.1 : 0.25, flexDirection: 'column', alignItems: 'flex-start' }}>
                    <ListItemText 
                      primary={template.template_name} 
                      primaryTypographyProps={{ sx: { fontSize: isMobile ? '0.8rem' : '0.875rem' }}}
                    />
                    {template.expansion && <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'text.secondary', fontSize: isMobile ? '0.7rem' : '0.75rem' }}>Expansion: {template.expansion}</Typography>}
                    {template.args && <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: isMobile ? '0.7rem' : '0.75rem' }}>Args: {JSON.stringify(template.args)}</Typography>}
                  </ListItem>
                ))}
              </List>
            </StyledAccordionDetails>
          </StyledAccordion>
        )}

        {/* Rhymes Section - Adjusted for mobile */}
        {rhymes.length > 0 && (
          <Box sx={{ 
              mt: templates.length > 0 || forms.length > 0 ? (isMobile ? 2 : 3) : 0, 
              pt: templates.length > 0 || forms.length > 0 ? (isMobile ? 1 : 2) : 0, 
              borderTop: templates.length > 0 || forms.length > 0 ? `1px solid ${theme.palette.divider}` : 'none',
              px: isMobile ? 1 : 0 // Add horizontal padding only on mobile for this box
            }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 500, mb: isMobile ? 1 : 1.5, fontSize: isMobile ? '0.85rem' : '0.875rem' }}>Rhymes</Typography>
            <List dense={isMobile} disablePadding>
              {rhymes.map((rhyme, index) => (
                <ListItem key={`rhyme-${index}`} disableGutters sx={{ py: isMobile ? 0.1 : 0.25 }}>
                  <ListItemText 
                    primary={rhyme.value} 
                    primaryTypographyProps={{ sx: { fontFamily: 'monospace', fontSize: isMobile ? '0.8rem' : '0.875rem'} }} 
                  />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
      </Box>
    );
  };

  const renderEtymologyTab = () => {
    // If the component is still loading the etymology data, show a spinner
    if (isLoadingEtymology) {
      return <Box sx={{ display: 'flex', justifyContent: 'center', p: isMobile ? 2 : 3 }}><CircularProgress /></Box>;
    }
    
    // If there was an error loading the etymology tree, show the error
    if (etymologyError) {
      return <Alert severity="error" sx={{ m: isMobile ? 1.5 : 2 }}>{etymologyError}</Alert>;
    }
    
    // First check if the word has self-contained etymology information
    const hasWordEtymologies = wordData.etymologies && wordData.etymologies.length > 0;
    const hasEtymologyTreeData = etymologyTree && etymologyTree.nodes && etymologyTree.nodes.length > 0;
    
    console.log("Etymology data check:", {
      wordId: wordData.id,
      word: wordData.lemma,
      hasWordEtymologies,
      etymologiesData: wordData.etymologies,
      etymologiesCount: wordData.etymologies?.length || 0,
      hasEtymologyTreeData,
      treeNodesCount: etymologyTree?.nodes?.length || 0
    });
    
    // Handle case where there's no etymology data from either source
    if (!hasWordEtymologies && !hasEtymologyTreeData) {
      return (
        <Box sx={{ p: isMobile ? 1.5 : 2 }}>
          <Alert severity="info" sx={{ mb: 2 }}>No etymology information available for word ID: {wordData.id} ({wordData.lemma}).</Alert>
          
          {/* Add links to external etymology resources */}
          <Typography variant="subtitle2" sx={{ mt: isMobile ? 2 : 3, mb: 1, fontSize: isMobile ? '0.85rem' : '0.875rem' }}>
            Try external etymology resources:
          </Typography>
          <Stack spacing={isMobile ? 0.5 : 1}>
            <Link 
              href={`https://en.wiktionary.org/wiki/${encodeURIComponent(wordData.lemma)}`} 
              target="_blank" 
              rel="noopener noreferrer"
              sx={{ fontSize: isMobile ? '0.8rem' : '0.875rem' }}
            >
              Look up "{wordData.lemma}" on Wiktionary
            </Link>
            {wordData.language_code === 'tl' && (
              <Link 
                href={`https://diksiyonaryo.ph/search/${encodeURIComponent(wordData.lemma)}`} 
                target="_blank" 
                rel="noopener noreferrer"
                sx={{ fontSize: isMobile ? '0.8rem' : '0.875rem' }}
              >
                Look up "{wordData.lemma}" on Diksiyonaryo.ph
              </Link>
            )}
          </Stack>
        </Box>
      );
    }
    
    // If there's etymology data in the word itself, display it regardless of tree
    if (hasWordEtymologies) {
      return (
        // Adjust padding for mobile
        <Box sx={{ p: isMobile ? 0 : theme.spacing(2) }}>
          {/* Display direct etymology data from word */}
          <List dense={isMobile}>
            {wordData.etymologies!.map((etym, index) => (
              <ListItem key={index} sx={{ 
                display: 'block', 
                py: isMobile ? 1 : 1.5,
                px: 0,
                borderBottom: index < wordData.etymologies!.length - 1 ? 
                  `1px solid ${theme.palette.divider}` : 'none'
              }}>
                <ListItemText
                  primary={
                    <Typography variant="subtitle1" sx={{ 
                      fontWeight: 600, 
                      mb: 0.5,
                      color: graphColors.main,
                      fontSize: isMobile ? '0.9rem' : '1rem'
                    }}>
                      Etymology {index + 1}
                    </Typography>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: isMobile ? 0.5 : 1 }}>
                      {/* Main text with improved styling */}
                      <Typography 
                        variant="body1" 
                        component="div" 
                        sx={{ 
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          lineHeight: isMobile ? 1.5 : 1.6, // Increased lineHeight
                          py: theme.spacing(isMobile ? 0.5 : 1),
                          fontSize: isMobile ? '0.85rem' : '0.875rem'
                        }}
                      >
                        {etym.text || etym.etymology_text}
                      </Typography>
                      
                      {/* Components with improved clickable styling */}
                      {etym.components && etym.components.length > 0 && (
                        <Box sx={{ mt: isMobile ? 0.5 : 1 }}>
                          <Typography 
                            variant="caption" 
                            component="div" 
                            color="text.secondary" 
                            sx={{ 
                              mb: 0.5,
                              fontWeight: 500, 
                              fontSize: isMobile ? '0.7rem' : '0.75rem'
                            }}
                          >
                            Components:
                          </Typography>
                          <Stack direction="row" spacing={isMobile ? 0.5 : 1} useFlexGap flexWrap="wrap">
                            {etym.components.map((comp, i) => (
                              <Chip 
                                key={i}
                                label={comp}
                                size="small"
                                clickable
                                onClick={() => onWordClick(comp)}
                                sx={{ 
                                  fontSize: isMobile ? '0.7rem' : '0.75rem',
                                  height: isMobile ? 20 : 24,
                                  bgcolor: alpha(graphColors.derived, 0.1),
                                  color: graphColors.derived,
                                  fontWeight: 500,
                                  '&:hover': {
                                    bgcolor: alpha(graphColors.derived, 0.2),
                                  }
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Languages with distinct styling */}
                      {etym.languages && etym.languages.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography 
                            variant="caption" 
                            component="div" 
                            color="text.secondary" 
                            sx={{ 
                              mb: 0.5,
                              fontWeight: 500
                            }}
                          >
                            Languages:
                          </Typography>
                          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                            {etym.languages.map((lang, i) => (
                              <Chip 
                                key={i}
                                label={lang}
                                size="small"
                                variant="outlined"
                                sx={{ 
                                  fontSize: '0.75rem',
                                  height: 24,
                                  borderColor: alpha(graphColors.variant, 0.6),
                                  color: graphColors.variant
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Sources with distinctive styling */}
                      {etym.sources && etym.sources.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography 
                            variant="caption" 
                            component="div" 
                            color="text.secondary" 
                            sx={{ 
                              mb: 0.5,
                              fontWeight: 500
                            }}
                          >
                            Sources:
                        </Typography>
                          <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                            {etym.sources.map((source, i) => (
                              <Chip 
                                key={i}
                                label={source}
                                size="small"
                                sx={{ 
                                  fontSize: '0.7rem',
                                  height: 'auto',
                                  padding: theme.spacing(0.25, 0),
                                  bgcolor: alpha(graphColors.related, 0.1),
                                  color: graphColors.related,
                                  '& .MuiChip-label': { 
                                    px: 1, 
                                    py: 0.25 
                                  }
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
                      )}
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
          
          {/* Show etymology tree data if available */}
          {hasEtymologyTreeData && (
            <Box sx={{ mt: 3, pt: 2, borderTop: `1px solid ${theme.palette.divider}` }}>
              <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>
                Etymology Tree
              </Typography>
              {renderEtymologyTreeVisualization()}
            </Box>
          )}
        </Box>
      );
    }
    
    // If only etymology tree data is available, render that
    return (
      <Box sx={{ p: 0 }}>
        {renderEtymologyTreeVisualization()}
      </Box>
    );
  };
  
  // Helper function to render the etymology tree visualization
  const renderEtymologyTreeVisualization = () => {
    // *** ADD NULL CHECKS ***
    if (!etymologyTree || !etymologyTree.nodes || !etymologyTree.edges) {
      // Optionally return a loading state or an informative message
      return <Alert severity="info">Etymology tree data is not available or is loading.</Alert>;
    }
  
    // Type definitions remain the same
    type EtymologyNode = { id: number; label: string; language?: string; [key: string]: any };
    type EtymologyEdge = { source: number; target: number; [key: string]: any };
    type EtymologyTreeMap = { [id: number]: EtymologyNode };
  
    const renderNode = (nodeId: number, nodes: EtymologyTreeMap, edges: EtymologyEdge[], level = 0): React.ReactNode => {
      const node = nodes[nodeId];
      if (!node) return null;
  
      const childrenEdges = edges.filter(edge => edge.source === nodeId);
  
      return (
        <Box key={node.id} sx={{ ml: level * 2.5, mb: 1.5 }}>
          <Paper elevation={1} sx={{ p: 1.5, display: 'flex', alignItems: 'center', gap: 1.5, borderRadius: '8px', bgcolor: alpha(getNodeColor(node.language || 'associated'), 0.1) }}>
            <Chip 
              label={node.language?.toUpperCase() || 'UNK'} 
              size="small" 
              sx={{ fontWeight: 600, bgcolor: getNodeColor(node.language || 'associated'), color: getTextColorForBackground(getNodeColor(node.language || 'associated')) }}
            />
            <Link
              component="button"
              onClick={() => onWordClick(`id:${node.id}`)} 
              sx={{ 
                fontWeight: 500, 
                fontSize: '1rem', 
                cursor: 'pointer', 
                color: 'text.primary', 
                '&:hover': { color: 'primary.main' }
              }}
            >
              {node.label}
            </Link>
          </Paper>
          {childrenEdges.length > 0 && (
            <Box sx={{ 
              mt: isMobile ? 0.5 : 1, 
              pl: isMobile ? 1.5 : 2, 
              // Refine borderLeft for visual connection
              borderLeft: `1px solid ${alpha(theme.palette.divider, isMobile ? 0.7 : 0.5)}` 
            }}>
              {childrenEdges.map(edge => renderNode(edge.target, nodes, edges, level + 1))}
            </Box>
          )}
        </Box>
      );
    };
    
    // *** Use etymologyTree safely after null check ***
    const rootNodes = etymologyTree.nodes.filter(node => 
      !etymologyTree.edges.some(edge => edge.target === node.id)
    );
    const nodeMap = etymologyTree.nodes.reduce((map, node) => {
      map[node.id] = node;
      return map;
    }, {} as EtymologyTreeMap);
    
    return (
      <Box sx={{ fontFamily: 'system-ui, sans-serif' }}>
        {rootNodes.map(rootNode => renderNode(rootNode.id, nodeMap, etymologyTree.edges))}
      </Box>
    );
  };

  // *** START: Re-insert renderSourcesInfoTab ***
  const renderSourcesInfoTab = () => {
     const credits = wordData.credits || [];
     const sourceInfo = wordData.source_info || {};
     const wordMetadata = wordData.word_metadata || {};
     const completeness = wordData.data_completeness || {};

     const hasCredits = credits.length > 0;
     const hasSourceInfo = Object.keys(sourceInfo).length > 0;
     const hasWordMeta = Object.keys(wordMetadata).length > 0;
     const hasCompleteness = Object.keys(completeness).length > 0;
     const hasEntryInfo = wordData.created_at || wordData.updated_at;

     if (!hasCredits && !hasSourceInfo && !hasWordMeta && !hasCompleteness && !hasEntryInfo) {
       return <Alert severity="info" sx={{ m: 2 }}>No source, metadata, or entry information available.</Alert>;
     }

     // Helper to render JSON data nicely
     const renderJsonData = (title: string, data: Record<string, any>) => {
       if (Object.keys(data).length === 0) return null;
       return (
         <StyledAccordion sx={{ mt: 2 }}>
           <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
             <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>{title}</Typography>
           </StyledAccordionSummary>
           <StyledAccordionDetails>
             <Paper variant="outlined" sx={{ p: 1.5, bgcolor: alpha(theme.palette.grey[500], 0.05), overflowX: 'auto' }}>
                <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                  {JSON.stringify(data, null, 2)}
                </Typography>
             </Paper>
           </StyledAccordionDetails>
         </StyledAccordion>
       );
     };

     return (
       <Box sx={{ p: theme.spacing(2) }}>
         {/* Credits List */}
         {hasCredits && (
           <>
             <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1.5 }}>Credits / Sources</Typography>
             <List dense sx={{ mb: theme.spacing(2) }}>
             {credits.map((credit, index) => (
                 <ListItem
                   key={credit.id || index}
                   sx={{
                     py: 1,
                     borderBottom: index < credits.length - 1 ?
                       `1px solid ${alpha(theme.palette.divider, 0.5)}` : 'none'
                   }}
                 >
                 <ListItemText
                     primary={<Typography variant="body1">{credit.credit}</Typography>}
                 />
               </ListItem>
             ))}
           </List>
          </>
         )}

         {/* Source Info JSON */}
         {renderJsonData('Source Info', sourceInfo)}

         {/* Word Metadata JSON */}
         {renderJsonData('Word Metadata', wordMetadata)}

         {/* Completeness Info */}
         {hasCompleteness && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>Data Completeness</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 1 }}>
                {Object.entries(completeness).map(([key, value]) => (
                  <Chip
                    key={key}
                    label={key.replace(/_/g, ' ')}
                    color={value ? "success" : "default"}
                    variant={value ? "filled" : "outlined"}
                    size="small"
                    sx={{ justifyContent: 'flex-start' }}
                  />
                ))}
              </Box>
            </Box>
         )}

         {/* Entry Timestamps */}
         {hasEntryInfo && (
           <Box sx={{ mt: 3, pt: 2, borderTop: hasCredits || hasSourceInfo || hasWordMeta || hasCompleteness ? `1px solid ${theme.palette.divider}` : 'none' }}>
             <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>Entry Information</Typography>
             <Stack spacing={1}>
               {wordData.created_at && (
                 <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                   <Typography variant="body2" color="text.secondary">Created:</Typography>
                   <Typography variant="body2">{new Date(wordData.created_at).toLocaleString()}</Typography>
                 </Box>
               )}
               {wordData.updated_at && (
                 <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                   <Typography variant="body2" color="text.secondary">Last Updated:</Typography>
                   <Typography variant="body2">{new Date(wordData.updated_at).toLocaleString()}</Typography>
                 </Box>
               )}
             </Stack>
           </Box>
         )}
       </Box>
     );
  };
  // *** END: Re-insert renderSourcesInfoTab ***

  // --- Main Component Return --- 
  if (!wordData?.id) {
    // Placeholder...
    return (
        <Paper elevation={1} sx={{ 
            p: 3, 
            height: '100%', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            bgcolor: isDarkMode ? 'var(--card-bg-color-elevated)' : 'background.default',
            color: isDarkMode ? 'var(--text-color-secondary)' : 'text.secondary'
        }}>
            <Typography color={isDarkMode ? 'inherit' : 'text.secondary'}>Select a word to see details.</Typography>
        </Paper>
    );
  }
            
  if (isLoading && !wordData?.id) { 
    // Loading indicator...
    return (
        <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', p: 3 }}>
            <CircularProgress />
            <Typography sx={{ ml: 2 }} color="text.secondary">Loading details...</Typography>
        </Box>
    );
  }
            
  // Main Render Logic
  return (
    <Box 
      ref={internalScrollRef} // Assign ref if needed for internal scroll logic
      sx={{
        height: '100%', // Ensure Box fills the container from Panel
        overflowY: 'auto', // Allow internal scrolling if content exceeds height
        overflowX: 'hidden',
         bgcolor: 'background.default', // Or appropriate background
         // Apply padding *inside* this Box now
         p: isMobile ? 1 : 2,
      }}
    >
      {/* Header Section */} 
      <Box sx={{ 
          p: isMobile ? 1.5 : 0, // Desktop padding is 0
          borderBottom: 1, 
          borderColor: 'divider',
          flexShrink: 0 
        }}>
        {renderHeader()}
      </Box>

      {/* Main Content Area (Tabs + Panel) */} 
      <Box sx={{ display: 'flex', flexGrow: 1, overflow: 'hidden' /* REMOVED height: '100%' */ }}>
        
        {/* Vertical Tabs (Desktop) */} 
        {!isMobile && (
          <Tabs
            orientation="vertical"
            variant="scrollable"
            value={activeTab}
            onChange={handleTabChange}
            aria-label="Word details sections"
            sx={{
              borderRight: 1,
              borderColor: 'divider',
              minWidth: 160, 
              bgcolor: 'background.paper',
              flexShrink: 0,
               '& .MuiTab-root': {
                    textTransform: 'none',
                    fontWeight: theme.typography.fontWeightRegular,
                    fontSize: theme.typography.pxToRem(14),
                    minHeight: 48, 
                    justifyContent: 'flex-start',
                    pl: 2, 
                    '&.Mui-selected': {
                      fontWeight: theme.typography.fontWeightMedium,
                      color: 'primary.main',
                    },
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    },
                  },
                  '& .MuiTabs-indicator': {
                    left: 0, 
                    width: 3, 
                    borderRadius: '3px 3px 0 0',
                  },
            }}
          >
            <Tab label="Definitions" value="definitions" />
            <Tab label="Relations" value="relations" />
            <Tab label="Etymology" value="etymology" />
            <Tab label="Forms" value="forms" />
            <Tab label="Sources" value="sources" />
          </Tabs>
        )} 

        {/* Tab Content Panel (Scrollable) */} 
        <Box
          role="tabpanel"
          hidden={false} // Keep it always rendered for simplicity
          sx={{
            flexGrow: 1, 
            overflowY: 'auto', 
            p: isMobile ? 1.5 : 2, // Apply padding here (desktop reduced)
            width: '100%', 
            minWidth: 0, 
          }}
        >
          {/* Conditionally render content based on activeTab */} 
          {activeTab === 'definitions' && renderDefinitionsTab()}
          {activeTab === 'relations' && renderRelationsTab()}
          {activeTab === 'etymology' && renderEtymologyTab()}
          {activeTab === 'forms' && renderFormsAndTemplatesTab()}
          {activeTab === 'sources' && renderSourcesInfoTab()}
        </Box>

      </Box> { /* End Main Content Area Box */}

      {/* Horizontal Tabs (Mobile) */} 
      {isMobile && (
        <Paper 
          square 
          sx={{ 
            borderTop: 1, 
            borderColor: 'divider', 
            flexShrink: 0, 
            // Explicitly set background and minimum height for visibility
            bgcolor: 'background.paper', // Ensure background color is applied
            minHeight: 48, // Give it a minimum height like before
            overflow: 'hidden' // Prevent children overflowing if sizing is wrong
          }}
        >
          <Tabs
            value={activeTab}
            onChange={handleTabChange}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            aria-label="Word details sections mobile"
            sx={{ 
              minHeight: 48, // Ensure Tabs tries to fill the Paper height
              width: '100%', // Ensure Tabs takes full width
              '& .MuiTab-root': { 
                fontSize: '0.75rem', 
                minWidth: 'auto', 
                p: 1,
                opacity: 1, // Ensure tabs are not transparent
                color: 'text.secondary', // Set a default color
                '&.Mui-selected': { // Ensure selected tab is clearly visible
                    color: 'primary.main', 
                    fontWeight: 'bold'
                }
              },
            }}
          >
            <Tab label="Defs" value="definitions" />
            <Tab label="Rels" value="relations" />
            <Tab label="Etym" value="etymology" />
            <Tab label="Forms" value="forms" />
            <Tab label="Srcs" value="sources" />
          </Tabs>
        </Paper>
      )}
    </Box> // End Main Wrapper Box
  );
}); // End forwardRef

// Export the component with memoization
export default React.memo(WordDetailsComponent);