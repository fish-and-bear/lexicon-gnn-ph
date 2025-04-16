import React, { useCallback, useState, useEffect } from 'react';
import { Definition, WordInfo, RelatedWord, NetworkLink, NetworkNode, WordForm, WordTemplate, Idiom, Affixation, DefinitionCategory, DefinitionLink, DefinitionRelation } from '../types';
// import { convertToBaybayin } from '../api/wordApi';
// import './WordDetails.css';
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
}));

const ExpandMoreIcon = () => <Typography sx={{ transform: 'rotate(90deg)', lineHeight: 0, color: 'text.secondary' }}>▶</Typography>;
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'primary.main' }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'error.main' }}>⏹️</Typography>;

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  etymologyTree,
  isLoadingEtymology,
  etymologyError,
  onWordLinkClick,
  onEtymologyNodeClick
}) => {
  const theme = useTheme(); // Get theme object
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md')); // Use 'md' breakpoint for vertical tabs
  const isDarkMode = theme.palette.mode === 'dark';

  const [tabValue, setTabValue] = useState<number>(0); // Tab value state
  const [activeTab, setActiveTab] = useState<string>('definitions'); // Use string for tab value
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  
  // Add these variables to fix the undefined errors
  const isLoading = false;
  const error = null;
  const onRetry = null;
  const drawerWidth = 280;

  // Effect to setup audio element
  useEffect(() => {
    setIsAudioPlaying(false); // Stop previous audio on word change
    const audioPronunciation = wordInfo?.pronunciations?.find(p => p.type === 'audio' && p.value);
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
  }, [wordInfo]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
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
    const ipaPronunciation = wordInfo.pronunciations?.find(p => p.type === 'IPA');
    const hasAudio = wordInfo.pronunciations?.some(p => p.type === 'audio' && p.value);
    const tags = wordInfo.tags ? wordInfo.tags.split(',').map(tag => tag.trim()).filter(Boolean) : [];

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

    const afterStyles = isDarkMode ? {
      top: 0,
      height: '2px',
      background: `linear-gradient(90deg, ${alpha(graphColors.main, 0.3)}, ${alpha(graphColors.main, 0)})`
    } : {
      bottom: 0,
      height: '3px',
      background: `linear-gradient(90deg, ${alpha(graphColors.main, 0.1)}, ${alpha(graphColors.main, 0)})`
    };

    return (
      // More elegant header with subtle styling
      <Box sx={{ 
        bgcolor: isDarkMode ? 'rgba(30, 40, 60, 0.4)' : headerBgColor, 
        color: headerTextColor, 
        pt: theme.spacing(3), 
        pb: theme.spacing(3), 
        pl: theme.spacing(3), 
        pr: theme.spacing(2),
        boxShadow: isDarkMode ? 'none' : 'inset 0 -1px 0 rgba(0,0,0,0.08)',
        borderBottom: isDarkMode ? '1px solid rgba(255,255,255,0.05)' : `1px solid ${alpha(theme.palette.divider, 0.08)}`,
        borderRadius: theme.spacing(0),
        position: 'relative',
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
        <Stack direction="row" spacing={1} alignItems="flex-start" flexWrap="nowrap" sx={{ mb: theme.spacing(1.5), width: '100%' }}>
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
              // Responsive font sizing
              fontSize: {
                xs: '1.5rem',   // Mobile
                sm: '1.75rem',  // Tablet 
                md: '2rem',     // Small desktop
                lg: '2.125rem', // Large desktop
              },
              // Additional styling
              fontWeight: 700, 
              letterSpacing: '-0.01em',
              color: isDarkMode ? '#ffffff' : theme.palette.text.primary,
              position: 'relative', // For pseudo-element
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: { xs: -2, sm: -4 }, 
                left: 0,
                width: '40px',
                height: '2px',
                bgcolor: mainColor,
                display: 'block',
                borderRadius: '2px',
              }
            }}
          >
            {wordInfo.lemma}
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

        {/* Pronunciation (IPA) - Enhanced styling */}
        {ipaPronunciation && (
          <Typography 
            variant="h6" 
            sx={{ 
              fontStyle: 'italic', 
              mb: theme.spacing(1.5), 
              pl: theme.spacing(0.5),
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

        {/* Baybayin - Show only if available in the database */}
        {wordInfo.has_baybayin && wordInfo.baybayin_form && wordInfo.baybayin_form.trim() !== '' && (
          <Box sx={{ my: theme.spacing(2) }}>
            <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.75), display: 'block', mb: 0.5 }}>
              Baybayin Script
            </Typography>
            <div 
              className="baybayin-text"
              style={{
                fontFamily: "'Noto Sans Baybayin', 'Arial Unicode MS', 'Noto Sans', sans-serif !important",
                fontSize: '2rem',
                padding: '8px 12px',
                background: isDarkMode ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.5)',
                borderRadius: '4px',
                display: 'inline-block',
                lineHeight: 1.2,
                minHeight: '48px',
                marginTop: '4px',
                color: 'inherit',
                border: isDarkMode ? '1px solid rgba(255, 255, 255, 0.05)' : '1px solid rgba(0, 0, 0, 0.05)'
              }}
            >
              {wordInfo.baybayin_form}
            </div>
          </Box>
        )}

        {/* Additional info - romanized form, language - Enhanced styling */}
        <Stack 
          direction="row" 
          spacing={3} 
          sx={{ 
            mt: 3,
            pt: 2,
            borderTop: `1px solid ${isDarkMode ? alpha('#fff', 0.05) : alpha('#000', 0.03)}`,
          }}
        >
          {wordInfo.language_code && (
            <Box>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(mainColor, 0.8),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.5 
                }}
              >
                Language
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 600,
                  display: 'inline-block',
                  px: 1,
                  py: 0.25,
                  borderRadius: '3px',
                  bgcolor: isDarkMode ? alpha(mainColor, 0.15) : alpha(mainColor, 0.08),
                  color: isDarkMode ? alpha(headerTextColor, 0.9) : mainColor,
                  border: `1px solid ${isDarkMode ? alpha(mainColor, 0.2) : alpha(mainColor, 0.15)}`
                }}
              >
                {wordInfo.language_code.toUpperCase()}
              </Typography>
            </Box>
          )}
          
          {wordInfo.romanized_form && (
            <Box>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(theme.palette.text.primary, 0.7),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.5 
                }}
              >
                Romanized Form
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 500,
                  color: isDarkMode ? alpha(headerTextColor, 0.95) : theme.palette.text.primary
                }}
              >
                {wordInfo.romanized_form}
              </Typography>
            </Box>
          )}
          
          {wordInfo.created_at && (
            <Box>
              <Typography 
                variant="caption" 
                sx={{ 
                  color: isDarkMode ? alpha(headerTextColor, 0.7) : alpha(theme.palette.text.primary, 0.7),
                  fontWeight: 500,
                  display: 'block', 
                  mb: 0.5 
                }}
              >
                Added
              </Typography>
              <Typography 
                variant="body2" 
                sx={{ 
                  fontWeight: 500,
                  color: isDarkMode ? alpha(headerTextColor, 0.95) : theme.palette.text.primary
                }}
              >
                {new Date(wordInfo.created_at).toLocaleDateString()}
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
    if (!wordInfo.definitions || wordInfo.definitions.length === 0) {
      return <Alert severity="info">No definitions available for this word.</Alert>;
    }

    // Group definitions first by POS, then by source
    const definitionsByPosThenSource: { [pos: string]: { [source: string]: Definition[] } } = {};
    
    wordInfo.definitions.forEach((def: Definition) => {
      const posKey = def.part_of_speech?.name_en || 'Other';
      // Handle cases where sources might be null or empty
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
      <Box sx={{ pt: theme.spacing(1), width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
        {Object.entries(definitionsByPosThenSource).map(([posName, defsBySource]) => (
          <Box key={posName} sx={{ mb: theme.spacing(3), width: '100%', maxWidth: '100%' }}>
            {/* Part of Speech Header */}
            <Typography 
              variant="subtitle1" 
              component="h3" 
              sx={{ 
                color: graphColors.main, 
                fontWeight: 600,
                pb: theme.spacing(1),
                borderBottom: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
                mb: theme.spacing(1.5),
                width: '100%',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
            >
              {posName}
            </Typography>
            
            {/* Iterate through definitions grouped by source */}
            {Object.entries(defsBySource).map(([sourceName, defs]) => (
              <Box key={sourceName} sx={{ mb: theme.spacing(2), width: '100%', maxWidth: '100%' }}> {/* Add margin between source groups */}
                <List disablePadding>
                  {defs.map((def: Definition, index: number) => {
                    const isLastDefinitionForSource = index === defs.length - 1;
                    return (
                    <ListItem 
                      key={def.id || index} 
                      alignItems="flex-start" 
                      sx={{ 
                        flexDirection: 'column', 
                        gap: 0.5, 
                        py: theme.spacing(1.5), 
                        pl: 0, 
                        position: 'relative', // Needed for absolute positioning of the chip
                        pb: isLastDefinitionForSource && sourceName !== 'Unknown Source' ? theme.spacing(3) : 1.5, // Add extra padding if chip is present
                      }}
                    >
                      {/* Definition text */}
                      <ListItemText
                        primaryTypographyProps={{ 
                          variant: 'body1', 
                          fontWeight: 500 
                        }}
                        primary={def.text}
                      />
                      
                      {/* Examples with quote styling */}
                      {def.examples && def.examples.length > 0 && (
                        <Box sx={{ pl: theme.spacing(2), mb: theme.spacing(1) }}>
                          {def.examples.map((example, exIndex) => (
                            <Typography 
                              key={exIndex} 
                              variant="body2" 
                              sx={{ 
                                fontStyle: 'italic', 
                                color: 'text.secondary',
                                mb: exIndex < def.examples.length - 1 ? 0.5 : 0,
                                position: 'relative',
                                pl: theme.spacing(3),
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
                      
                      {/* Usage notes if available */}
                      {def.usage_notes && def.usage_notes.length > 0 && (
                        <Box sx={{ pl: theme.spacing(1), mb: theme.spacing(1) }}>
                          <Typography 
                            variant="caption" 
                            component="div" 
                            sx={{ fontWeight: 500, mb: 0.5 }}
                          >
                            Usage Notes:
                          </Typography>
                          {def.usage_notes.map((note, noteIndex) => (
                            <Typography 
                              key={noteIndex} 
                              variant="body2" 
                              sx={{ color: 'text.secondary' }}
                            >
                              {note}
                            </Typography>
                          ))}
                        </Box>
                      )}
                      
                      {/* Definition metadata/tags if available */}
                      {def.tags && def.tags.length > 0 && (
                        <Stack 
                          direction="row" 
                          spacing={0.5} 
                          useFlexGap 
                          flexWrap="wrap" 
                          sx={{ mt: theme.spacing(0.5) }}
                        >
                          {def.tags.map((tag, tagIndex) => (
                            <Chip
                              key={tagIndex}
                              label={tag}
                              size="small"
                              sx={{
                                fontSize: '0.7rem',
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

                      {/* Display Source Chip only once after the LAST definition, bottom right */}
                      {isLastDefinitionForSource && sourceName !== 'Unknown Source' && (
                        <Chip
                          label={`${sourceName}`}
                          size="small"
                          variant="outlined"
                          sx={{
                            position: 'absolute', // Position relative to ListItem
                            bottom: theme.spacing(0.5),
                            right: theme.spacing(0.5),
                            fontSize: '0.7rem',
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
                  );})}
                </List>
              </Box>
            ))}
          </Box>
        ))}
      </Box>
    );
  };

  const renderRelationsTab = () => {
    // Check for defined relations or default to empty arrays
    const incoming_relations = wordInfo.incoming_relations || [];
    const outgoing_relations = wordInfo.outgoing_relations || [];
    
    // Get semantic network data safely
    const semanticNetworkNodes = wordInfo.semantic_network?.nodes || [];
    const semanticNetworkLinks = wordInfo.semantic_network?.links || [];
    
    // Enhanced fallback logic: Use semantic network if regular relations are empty OR there's an error message
    // This handles the case where the backend returns an error due to the missing relation_data column
    const hasError = typeof wordInfo.server_error === 'string' && wordInfo.server_error.includes('database error');
    const useSemanticNetwork = 
      ((incoming_relations.length === 0 && outgoing_relations.length === 0) || hasError) && 
      semanticNetworkLinks.length > 0;
    
    // Log the fallback status
    if (hasError) {
      console.log("Using semantic network fallback due to server error:", wordInfo.server_error);
    }
      
    // Helper function to create relation objects from semantic network data
    function createRelationsFromNetwork() {
      console.log("Creating relations from semantic network as fallback");
      const mainWord = wordInfo.lemma;
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
    const hasAnyRelations = hasStandardRelations || semanticRelations.length > 0 || (wordInfo.root_affixations && wordInfo.root_affixations.length > 0) || (wordInfo.affixed_affixations && wordInfo.affixed_affixations.length > 0);
      
    if (!hasAnyRelations) {
        return <Alert severity="info">No relations or affixations available for this word.</Alert>;
    }

    // Helper to render Affixations Section
    const renderAffixations = () => {
      const rootAffix = wordInfo.root_affixations || [];
      const affixedAffix = wordInfo.affixed_affixations || [];

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
                        primaryTypographyProps={{ component: 'span', variant: 'body2', sx: { cursor: 'pointer', textDecoration: 'underline', color: theme.palette.primary.main, mr: 1 }, onClick: () => affix.affixed_word && onWordLinkClick(affix.affixed_word.lemma) }}
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
                        primaryTypographyProps={{ component: 'span', variant: 'body2', sx: { cursor: 'pointer', textDecoration: 'underline', color: theme.palette.primary.main, mr: 1 }, onClick: () => affix.root_word && onWordLinkClick(affix.root_word.lemma) }}
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
      return rel.wordObj?.lemma !== wordInfo.lemma;
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
            {wordInfo.server_error || 'Server database error. Using semantic network relationships as fallback.'}
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
                            spacing={1} 
                            useFlexGap 
                            flexWrap="wrap"
                            sx={{ mb: theme.spacing(1.5), pl: theme.spacing(1), width: '100%', maxWidth: '100%' }}
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
                                  onClick={() => onWordLinkClick(wordObj.lemma || wordObj.word)}
                                  variant="outlined"
                                  sx={{
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
                                    '&:hover': {
                                      backgroundColor: alpha(relColor, 0.1),
                                      borderColor: relColor
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
        {wordInfo.data_completeness && (
          <Box sx={{ mt: theme.spacing(3), pt: theme.spacing(2), borderTop: `1px solid ${theme.palette.divider}` }}>
            <Typography variant="subtitle2" sx={{ mb: theme.spacing(1) }}>
              Data Completeness
            </Typography>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 1 }}>
              {Object.entries(wordInfo.data_completeness).map(([key, value]) => (
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
    const forms = wordInfo.forms || [];
    const templates = wordInfo.templates || [];

    if (forms.length === 0 && templates.length === 0) {
      return <Alert severity="info">No forms or templates available.</Alert>;
    }

    return (
      <Box sx={{ pt: theme.spacing(1) }}>
        {/* Forms Section */}
        {forms.length > 0 && (
          <StyledAccordion defaultExpanded sx={{ mb: 2 }}>
            <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Word Forms (Inflections/Conjugations)</Typography>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              <List dense disablePadding>
                {forms.map((form: WordForm, index: number) => (
                  <ListItem key={form.id || index} disableGutters sx={{ py: 0.25 }}>
                    <ListItemText primary={form.form} />
                    <Stack direction="row" spacing={0.5}>
                       {form.is_canonical && <Chip label="Canonical" size="small" color="primary" variant="outlined" sx={{ height: 'auto', fontSize: '0.6rem' }} />}
                       {form.is_primary && <Chip label="Primary" size="small" color="secondary" variant="outlined" sx={{ height: 'auto', fontSize: '0.6rem' }} />}
                       {form.tags && Object.entries(form.tags).map(([key, value]) => (
                          <Chip key={key} label={`${key}: ${value}`} size="small" sx={{ height: 'auto', fontSize: '0.6rem' }} />
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
              <Typography variant="subtitle2" sx={{ fontWeight: 500 }}>Word Templates</Typography>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              <List dense disablePadding>
                {templates.map((template: WordTemplate, index: number) => (
                  <ListItem key={template.id || index} disableGutters sx={{ py: 0.25, flexDirection: 'column', alignItems: 'flex-start' }}>
                    <ListItemText primary={template.template_name} />
                    {template.expansion && <Typography variant="caption" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>Expansion: {template.expansion}</Typography>}
                    {template.args && <Typography variant="caption" sx={{ color: 'text.secondary' }}>Args: {JSON.stringify(template.args)}</Typography>}
                  </ListItem>
                ))}
              </List>
            </StyledAccordionDetails>
          </StyledAccordion>
        )}
      </Box>
    );
  };

  const renderEtymologyTab = () => {
    // If the component is still loading the etymology data, show a spinner
    if (isLoadingEtymology) {
      return <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}><CircularProgress /></Box>;
    }
    
    // If there was an error loading the etymology tree, show the error
    if (etymologyError) {
      return <Alert severity="error" sx={{ m: 2 }}>{etymologyError}</Alert>;
    }
    
    // First check if the word has self-contained etymology information
    const hasWordEtymologies = wordInfo.etymologies && wordInfo.etymologies.length > 0;
    const hasEtymologyTreeData = etymologyTree && etymologyTree.nodes && etymologyTree.nodes.length > 0;
    
    console.log("Etymology data check:", {
      wordId: wordInfo.id,
      word: wordInfo.lemma,
      hasWordEtymologies,
      etymologiesData: wordInfo.etymologies,
      etymologiesCount: wordInfo.etymologies?.length || 0,
      hasEtymologyTreeData,
      treeNodesCount: etymologyTree?.nodes?.length || 0
    });
    
    // Handle case where there's no etymology data from either source
    if (!hasWordEtymologies && !hasEtymologyTreeData) {
      return (
        <Box sx={{ p: 2 }}>
          <Alert severity="info" sx={{ mb: 2 }}>No etymology information available for word ID: {wordInfo.id} ({wordInfo.lemma}).</Alert>
          
          {/* Add links to external etymology resources */}
          <Typography variant="subtitle2" sx={{ mt: 3, mb: 1 }}>
            Try external etymology resources:
          </Typography>
          <Stack spacing={1}>
            <Link 
              href={`https://en.wiktionary.org/wiki/${encodeURIComponent(wordInfo.lemma)}`} 
              target="_blank" 
              rel="noopener noreferrer"
            >
              Look up "{wordInfo.lemma}" on Wiktionary
            </Link>
            {wordInfo.language_code === 'tl' && (
              <Link 
                href={`https://diksiyonaryo.ph/search/${encodeURIComponent(wordInfo.lemma)}`} 
                target="_blank" 
                rel="noopener noreferrer"
              >
                Look up "{wordInfo.lemma}" on Diksiyonaryo.ph
              </Link>
            )}
          </Stack>
        </Box>
      );
    }
    
    // If there's etymology data in the word itself, display it regardless of tree
    if (hasWordEtymologies) {
      return (
        <Box sx={{ p: theme.spacing(2) }}>
          {/* Display direct etymology data from word */}
          <List dense>
            {wordInfo.etymologies!.map((etym, index) => (
              <ListItem key={index} sx={{ 
                display: 'block', 
                py: 1.5,
                px: 0,
                borderBottom: index < wordInfo.etymologies!.length - 1 ? 
                  `1px solid ${theme.palette.divider}` : 'none'
              }}>
                <ListItemText
                  primary={
                    <Typography variant="subtitle1" sx={{ 
                      fontWeight: 600, 
                      mb: 0.5,
                      color: graphColors.main
                    }}>
                      Etymology {index + 1}
                    </Typography>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {/* Main text with improved styling */}
                      <Typography 
                        variant="body1" 
                        component="div" 
                        sx={{ 
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          lineHeight: 1.5,
                          py: theme.spacing(1)
                        }}
                      >
                        {etym.text || etym.etymology_text}
                      </Typography>
                      
                      {/* Components with improved clickable styling */}
                      {etym.components && etym.components.length > 0 && (
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
                            Components:
                          </Typography>
                          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                            {etym.components.map((comp, i) => (
                              <Chip 
                                key={i}
                                label={comp}
                                size="small"
                                clickable
                                onClick={() => onWordLinkClick(comp)}
                                sx={{ 
                                  fontSize: '0.75rem',
                                  height: 24,
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
    // Assume Etymology Tree structure
    type EtymologyNode = { id: number; label: string; language?: string; [key: string]: any };
    type EtymologyEdge = { source: number; target: number; [key: string]: any };
    type EtymologyTreeMap = { [id: number]: EtymologyNode };

    // Basic List Rendering of Etymology Nodes
    const renderNode = (nodeId: number, nodes: EtymologyTreeMap, edges: EtymologyEdge[], level = 0): React.ReactNode => {
       const node = nodes[nodeId];
       if (!node) return null;

       // Find children (nodes where the current node is a source)
       const childrenEdges = edges.filter((edge: EtymologyEdge) => edge.source === nodeId);
       const childrenIds = childrenEdges.map((edge: EtymologyEdge) => edge.target);

       return (
          <ListItem key={node.id} sx={{ pl: level * 2.5, display: 'block', py: 0.5, borderLeft: level > 0 ? `1px dashed ${theme.palette.divider}` : 'none', ml: level > 0 ? 0.5 : 0 }}> {/* Indentation based on level */}
             <ListItemText
                primary={
                    <Typography 
                      variant="body2" 
                      component="span" 
                      sx={{ 
                        fontWeight: level === 0 ? 600 : 400,
                        cursor: 'pointer',
                        '&:hover': { textDecoration: 'underline' }
                      }}
                      onClick={() => onEtymologyNodeClick(node.id)}
                    >
                        {node.label}
                    </Typography>
                }
                secondary={node.language ? `(${node.language})` : null}
                sx={{ my: 0 }}
             />
             {childrenIds.length > 0 && (
                <List dense disablePadding sx={{ pl: 0 }}>
                   {childrenIds.map(childId => renderNode(childId, nodes, edges, level + 1))}
                </List>
             )}
          </ListItem>
       );
    };

    // Find the root node(s) - nodes that are not targets of any edge
    const targetIds = new Set(etymologyTree.edges.map((edge: EtymologyEdge) => edge.target));
    const rootIds = etymologyTree.nodes
                      .filter((node: EtymologyNode) => !targetIds.has(node.id))
                      .map((node: EtymologyNode) => node.id);

    // Build a map for quick node lookup
    const nodeMap = etymologyTree.nodes.reduce((acc: EtymologyTreeMap, node: EtymologyNode) => {
        acc[node.id] = node;
        return acc;
    }, {});

    return (
      <List dense sx={{ pt: 1 }}>
         {rootIds.length > 0
            ? rootIds.map((rootId: number) => renderNode(rootId, nodeMap, etymologyTree.edges))
            : <ListItem><Alert severity="warning" variant="outlined" sx={{ width: '100%' }}>Could not determine root etymology node.</Alert></ListItem> }
      </List>
    );
  };

  const renderSourcesInfoTab = () => {
     const credits = wordInfo.credits || [];
     const sourceInfo = wordInfo.source_info || {};
     const wordMetadata = wordInfo.word_metadata || {};
     const completeness = wordInfo.data_completeness || {};

     const hasCredits = credits.length > 0;
     const hasSourceInfo = Object.keys(sourceInfo).length > 0;
     const hasWordMeta = Object.keys(wordMetadata).length > 0;
     const hasCompleteness = Object.keys(completeness).length > 0;
     const hasEntryInfo = wordInfo.created_at || wordInfo.updated_at;

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
             <Paper variant="outlined" sx={{ p: 1.5, bgcolor: alpha(theme.palette.grey[500], 0.05), overflowX: 'auto' }}> {/* Add horizontal scroll as fallback */} 
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
               {wordInfo.created_at && (
                 <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                   <Typography variant="body2" color="text.secondary">Created:</Typography>
                   <Typography variant="body2">{new Date(wordInfo.created_at).toLocaleString()}</Typography>
                 </Box>
               )}
               {wordInfo.updated_at && (
                 <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                   <Typography variant="body2" color="text.secondary">Last Updated:</Typography>
                   <Typography variant="body2">{new Date(wordInfo.updated_at).toLocaleString()}</Typography>
                 </Box>
               )}
             </Stack>
           </Box>
         )}
       </Box>
     );
  };

  // --- Main Component Return ---
  if (!wordInfo?.id) {
    // Render a placeholder or empty state if no word is selected
    return (
        <Paper elevation={1} sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'background.default' }}>
            <Typography color="text.secondary">Select a word to see details.</Typography>
        </Paper>
              );
            }
            
            const tabContent = (
               <Box sx={{ 
                  flexGrow: 1, 
                  overflowY: 'auto', 
                  position: 'relative', 
                  bgcolor: 'background.default',
                  ...(isWideScreen && { mt: 0 })
               }}>
                  <Box sx={{ px: theme.spacing(3), py: theme.spacing(2) }}>
                      {activeTab === 'definitions' && renderDefinitionsTab()}
                      {activeTab === 'relations' && renderRelationsTab()}
                      {activeTab === 'forms_templates' && renderFormsAndTemplatesTab()}
                      {activeTab === 'etymology' && renderEtymologyTab()}
                      {activeTab === 'sources-info' && renderSourcesInfoTab()}
                      {activeTab === 'metadata' && renderSourcesInfoTab()}
                  </Box>
               </Box>
            );

            const tabs = (
               <Box sx={{ height: '100%' }}>
                 <Tabs
                   onChange={handleTabChange}
                   value={activeTab}
                   aria-label="Word details tabs"
                   orientation={isWideScreen ? 'vertical' : 'horizontal'}
                   variant="scrollable"
                   sx={{
                     borderRight: isWideScreen ? 0 : 0,
                     borderBottom: isWideScreen ? 0 : 1,
                     borderColor: 'divider',
                     minWidth: isWideScreen ? 160 : 'auto',
                     alignItems: isWideScreen ? 'flex-start' : 'center',
                     margin: 0,
                     padding: 0,
                     height: '100%',
                     '& .MuiTab-root': { 
                       minHeight: 48, 
                       alignItems: 'flex-start',
                       textAlign: 'left',
                       px: isWideScreen ? 2 : 1,
                     },
                   }}
                 >
                   {/* Tab Definitions */}
                   <Tab label="Definitions" value="definitions" />
                   {/* Tab Relations */}
                   <Tab label="Relations" value="relations" />
                   {/* Tab Forms & Templates */}
                   <Tab label="Forms & Templates" value="forms_templates" />
                   {/* Tab Etymology */}
                   <Tab label="Etymology" value="etymology" />
                   {/* Tab Metadata - Only show if data exists */}
                   {(wordInfo.source_info || wordInfo.word_metadata || wordInfo.tags) && (
                     <Tab label="Metadata" value="metadata" />
                   )}
                 </Tabs>
               </Box>
            );

  return (
    <Paper 
      elevation={2} 
      square 
      sx={{ 
        display: 'flex', 
        flexDirection: 'column', 
        height: '100%', 
        width: '100%', 
        bgcolor: 'background.paper',
        overflow: 'hidden',
        maxWidth: '100%'
      }}
    >

      {/* Conditional Layout based on screen size */}
      {isWideScreen ? (
          <>
            {renderHeader()}
            <Box sx={{ 
              flexGrow: 1, 
              display: 'flex', 
              flexDirection: 'row', 
              overflow: 'hidden',
              width: '100%'
            }}>
               <Box sx={{ 
                 display: 'flex', 
                 flexDirection: 'column',
                 minWidth: 160,
                 maxWidth: 160,
                 borderRight: 1,
                 borderColor: 'divider',
               }}>
                 {tabs}
               </Box>
               <Box sx={{ 
                 flexGrow: 1, 
                 overflowY: 'auto', 
                 position: 'relative', 
                 bgcolor: 'background.default',
                 width: 'calc(100% - 160px)',
                 maxWidth: '100%'
               }}>
                 <Box sx={{ px: theme.spacing(3), py: theme.spacing(2), width: '100%', maxWidth: '100%' }}>
                     {activeTab === 'definitions' && renderDefinitionsTab()}
                     {activeTab === 'relations' && renderRelationsTab()}
                     {activeTab === 'forms_templates' && renderFormsAndTemplatesTab()}
                     {activeTab === 'etymology' && renderEtymologyTab()}
                     {activeTab === 'sources-info' && renderSourcesInfoTab()}
                     {activeTab === 'metadata' && renderSourcesInfoTab()}
                 </Box>
               </Box>
            </Box>
          </>
      ) : (
          <>
            {renderHeader()}
            {tabs}
            {/* Ensure tabContent takes remaining space and scrolls in vertical layout */}
            <Box sx={{ 
               flexGrow: 1, 
               overflowY: 'auto', 
               minHeight: 0,
               position: 'relative',
               bgcolor: 'background.default',
               width: '100%'
            }}>
               <Box sx={{ px: theme.spacing(3), py: theme.spacing(2), width: '100%', maxWidth: '100%' }}>
                   {activeTab === 'definitions' && renderDefinitionsTab()}
                   {activeTab === 'relations' && renderRelationsTab()}
                   {activeTab === 'forms_templates' && renderFormsAndTemplatesTab()}
                   {activeTab === 'etymology' && renderEtymologyTab()}
                   {activeTab === 'sources-info' && renderSourcesInfoTab()}
                   {activeTab === 'metadata' && renderSourcesInfoTab()}
               </Box>
            </Box>
          </>
      )}
      {/* Add comment about resizing parent */}
      {/* For resizable sidebar functionality, the parent component (e.g., WordExplorer) needs layout adjustments (e.g., using SplitPane) */}
    </Paper>
  );
});

export default WordDetails;