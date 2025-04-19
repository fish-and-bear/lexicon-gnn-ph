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
  onWordLinkClick: (word: string | number) => void; // Allow number for IDs
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
  // --- DEBUG: Comment out unused props for now ---
  // etymologyTree,
  // isLoadingEtymology,
  // etymologyError,
  onWordLinkClick,
  // onEtymologyNodeClick
  // --- END DEBUG ---
}) => {
  const theme = useTheme();
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md'));
  const isDarkMode = theme.palette.mode === 'dark';

  // --- DEBUG: Comment out state and effects --- 
  /*
  const [activeTab, setActiveTab] = useState<string>('definitions'); // Default to definitions
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

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

  const playAudio = useCallback((url: string) => {
    if (audioElement && isAudioPlaying) {
      stopAudio(); // Stop current audio if playing
      return;
    }
    const newAudio = new Audio(url);
    setAudioElement(newAudio);
    newAudio.play().then(() => {
      setIsAudioPlaying(true);
      const onEnded = () => setIsAudioPlaying(false);
      newAudio.addEventListener('ended', onEnded);
      // Cleanup listener when component unmounts or audio changes
      return () => newAudio.removeEventListener('ended', onEnded);
    }).catch(error => {
      console.error("Error playing audio:", error);
      setAudioElement(null); // Reset if playback fails
    });
  }, [audioElement, isAudioPlaying]); // Add isAudioPlaying

  const stopAudio = useCallback(() => {
    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
      setIsAudioPlaying(false);
    }
  }, [audioElement]);
  */
  // --- END DEBUG ---
  
  // --- DEBUG: Keep only renderHeader and renderWordLink initially --- 
  const renderHeader = () => {
    if (!wordInfo) return null;
    return (
      <Box>
        <Typography variant="h4" component="h2" gutterBottom sx={{ color: 'var(--header-text-color)' }}>
          {wordInfo.lemma}
        </Typography>
        {wordInfo.romanized_form && wordInfo.romanized_form !== wordInfo.lemma && (
          <Typography variant="subtitle1" sx={{ color: 'var(--secondary-text-color)', fontStyle: 'italic' }} gutterBottom>
            Romanized: {wordInfo.romanized_form}
          </Typography>
        )}
        {wordInfo.language_code && (
          <Chip label={`Language: ${wordInfo.language_code.toUpperCase()}`} size="small" sx={{ mr: 1, bgcolor: 'var(--chip-bg-color)', color: 'var(--chip-text-color)' }} />
        )}
        {wordInfo.has_baybayin && (
           <Chip 
              label="Baybayin Available" 
              size="small" 
              sx={{ 
                 mr: 1, 
                 bgcolor: 'var(--baybayin-chip-bg)', 
                 color: 'var(--baybayin-chip-text)', 
                 fontWeight: 'bold' 
              }} 
           />
        )}
        {wordInfo.baybayin_form && (
          <Typography 
             variant="body1" 
             sx={{ 
                 fontFamily: '"Noto Sans Baybayin", sans-serif', // Ensure Baybayin font is applied
                 fontSize: '1.5rem', 
                 my: 1, 
                 color: 'var(--baybayin-color)'
             }}
          >
            ᜊᜌ᜔ᜊᜌᜒᜈ᜔: {wordInfo.baybayin_form}
          </Typography>
        )}
        {/* Pronunciation Buttons */} 
        {wordInfo.pronunciations && wordInfo.pronunciations.length > 0 && (
          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
            {wordInfo.pronunciations.map((pron, index) => {
              if (pron.type === 'audio_url' && pron.value) {
                return (
                  <IconButton 
                    key={`${pron.id || index}-audio`} 
                    onClick={() => playAudio(pron.value)} 
                    disabled={isAudioPlaying && audioElement?.src === pron.value}
                    color="primary"
                    size="small"
                    title={`Play audio pronunciation`}
                  >
                    {isAudioPlaying && audioElement?.src === pron.value ? <StopCircleIcon /> : <VolumeUpIcon />}
                  </IconButton>
                );
              } else if (pron.type === 'ipa' && pron.value) {
                return (
                  <Chip key={`${pron.id || index}-ipa`} label={`IPA: /${pron.value}/`} size="small" variant="outlined" sx={{ bgcolor: 'var(--chip-bg-color)', color: 'var(--chip-text-color)' }} />
                );
              }
              return null;
            })}
            {isAudioPlaying && (
              <Button size="small" onClick={stopAudio} color="secondary" startIcon={<StopCircleIcon />}>
                 Stop Audio
              </Button>
            )}
          </Stack>
        )}
        {/* Tags */} 
        {wordInfo.tags && (
          <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
            {wordInfo.tags.split(',').map((tag, index) => (
              <Chip key={index} label={tag.trim()} size="small" sx={{ bgcolor: 'var(--chip-bg-color)', color: 'var(--chip-text-color)' }} />
            ))}
          </Stack>
        )}
      </Box>
    );
  };

  const renderWordLink = (word: RelatedWord | { lemma: string, id?: number | string } | string, key?: React.Key): React.ReactNode => {
    let lemma: string;
    let id: number | string | undefined;

    if (typeof word === 'string') {
      lemma = word;
      id = word; // Use lemma itself as identifier if no ID
    } else {
      lemma = word.lemma;
      id = word.id ?? lemma; // Use ID if available, otherwise lemma
    }
    
    return (
      <Link
        key={key}
        component="button"
        onClick={() => onWordLinkClick(id!)} // Use non-null assertion for ID
        sx={{ 
          mx: 0.5, 
          fontWeight: 'medium', 
          color: 'var(--link-color)',
          textDecorationColor: 'var(--link-color)',
          '&:hover': {
            textDecorationColor: 'var(--accent-color)',
          }
        }}
      >
        {lemma}
      </Link>
    );
  };
  
  const renderSafeHtml = (htmlString: string | undefined | null): React.ReactNode => {
    if (!htmlString) return null;
    // Basic sanitization (consider DOMPurify for robust sanitization if needed)
    const sanitizedHtml = htmlString.replace(/<script.*?>.*?<\/script>/gi, '');
    return <span dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />;
  };

  // --- DEBUG: Comment out tab rendering functions --- 
  /*
  const renderDefinitionsTab = () => {
    if (!wordInfo || !wordInfo.definitions || wordInfo.definitions.length === 0) {
      return <Typography>No definitions available.</Typography>;
    }

    // Group definitions by Part of Speech
    const definitionsByPOS: Record<string, Definition[]> = {};
    wordInfo.definitions.forEach(def => {
      const posName = def.part_of_speech?.name_en || def.original_pos || 'Unclassified';
      if (!definitionsByPOS[posName]) {
        definitionsByPOS[posName] = [];
      }
      definitionsByPOS[posName].push(def);
    });

    return (
      <Box>
        {Object.entries(definitionsByPOS).map(([pos, definitions]) => (
          <DefinitionCard key={pos} sx={{ mb: 3 }}> {/* Add more bottom margin */}
            <Typography variant="h6" component="h3" gutterBottom sx={{ color: 'var(--section-header-color)' }}>
              {pos}
            </Typography>
            <List disablePadding>
              {definitions.map((def, index) => (
                <React.Fragment key={def.id || index}>
                  <ListItem alignItems="flex-start" disableGutters>
                    <ListItemText
                      primary={<Typography variant="body1" sx={{ color: 'var(--text-color)' }}>{renderSafeHtml(def.text)}</Typography>}
                      secondary={
                        <Stack spacing={1} sx={{ mt: 1 }}>
                          {def.examples && def.examples.length > 0 && (
                            <Box sx={{ pl: 2, borderLeft: `2px solid ${theme.palette.divider}`, my: 1 }}>
                              <Typography variant="body2" sx={{ fontStyle: 'italic', color: 'var(--secondary-text-color)' }}>
                                Example:
                              </Typography>
                              {def.examples.map((ex, i) => (
                                <Typography key={i} variant="body2" sx={{ fontStyle: 'italic', color: 'var(--secondary-text-color)', pl: 1 }}>
                                  - {renderSafeHtml(ex)}
                                </Typography>
                              ))}
                            </Box>
                          )}
                          {def.usage_notes && def.usage_notes.length > 0 && (
                            <Typography variant="caption" sx={{ color: 'var(--secondary-text-color)' }}>
                              Usage: {def.usage_notes.join('; ')}
                            </Typography>
                          )}
                          {/* Definition Relations */}
                          {def.definition_relations && def.definition_relations.length > 0 && (
                            <Box>
                              <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'var(--secondary-text-color)' }}>See also:</Typography>
                              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                                {def.definition_relations.map((rel, relIndex) => (
                                  rel.related_word && renderWordLink(rel.related_word, `${def.id}-rel-${relIndex}`)
                                ))}
                              </Stack>
                            </Box>
                          )}
                          {/* Tags */}
                          {def.tags && def.tags.length > 0 && (
                            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                              {def.tags.map((tag, tagIndex) => (
                                <Chip key={tagIndex} label={tag} size="small" variant="outlined" sx={{ bgcolor: 'var(--chip-bg-color)', color: 'var(--chip-text-color)' }}/>
                              ))}
                            </Stack>
                          )}
                          {/* Sources */}
                          {def.sources && def.sources.length > 0 && (
                            <SourceTag>
                              Source(s): {def.sources.join(', ')}
                            </SourceTag>
                          )}
                          {/* Categories */}
                          {def.categories && def.categories.length > 0 && (
                             <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{mt: 1}}>
                              <Typography variant="caption" sx={{fontWeight: 'bold'}}>Categories:</Typography>
                              {def.categories.map((cat, catIndex) => (
                                <Chip key={cat.id || catIndex} label={cat.category_name} size="small" variant="filled" sx={{ bgcolor: alpha(theme.palette.secondary.main, 0.2), color: 'secondary.dark' }}/>
                              ))}
                            </Stack>
                          )}
                          {/* Links */}
                          {def.links && def.links.length > 0 && (
                            <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{mt: 1}}>
                              <Typography variant="caption" sx={{fontWeight: 'bold'}}>Links:</Typography>
                              {def.links.map((link, linkIndex) => (
                                <Link key={link.id || linkIndex} href={link.target_url} target="_blank" rel="noopener noreferrer" variant="caption">
                                  {link.display_text || link.link_text || 'Link'}
                                </Link>
                              ))}
                            </Stack>
                          )}
                        </Stack>
                      }
                    />
                  </ListItem>
                  {index < definitions.length - 1 && <Divider component="li" variant="inset" />}
                </React.Fragment>
              ))}
            </List>
          </DefinitionCard>
        ))}
      </Box>
    );
  };
  
  const renderRelationsTab = () => {
    // ... implementation ...
  };

  const renderFormsAndTemplatesTab = () => {
    // ... implementation ...
  };

  const renderEtymologyTab = () => {
    // ... implementation ...
  };

  const renderSourcesInfoTab = () => {
    // ... implementation ...
  };
  */
  // --- END DEBUG --- 

  // --- DEBUG: Comment out dynamic tab calculation --- 
  /*
  const availableTabs = useMemo(() => {
    // ... implementation ...
  }, [wordInfo]); 
  */
  // --- END DEBUG ---

  // --- DEBUG: Comment out tab reset effect --- 
  /*
  useEffect(() => {
    // ... implementation ...
  }, [availableTabs, activeTab, wordInfo?.id]); 
  */
  // --- END DEBUG ---

  return (
    <Paper elevation={2} sx={{ 
        display: 'flex', 
        flexDirection: isWideScreen ? 'row' : 'column', 
        height: '100%', // Ensure it takes full height of its container
        overflow: 'hidden', // Prevent internal scrolling issues
        border: `1px solid ${theme.palette.divider}`,
        backgroundColor: isDarkMode ? '#2C2C2C' : '#f9f9f9'
      }}
    >
      {/* Header Section */}
      <Box sx={{ p: 2, borderBottom: !isWideScreen ? `1px solid ${theme.palette.divider}` : 'none', borderRight: isWideScreen ? `1px solid ${theme.palette.divider}` : 'none' }}>
        {renderHeader()}
      </Box>

      {/* --- DEBUG: Render simple placeholder instead of tabs --- */}
      <Box sx={{ p: 2, flexGrow: 1, overflowY: 'auto' }}>
        <Typography>Word Details Component (Simplified)</Typography>
        <Typography variant="caption">Tabs and complex rendering disabled.</Typography>
      </Box>
      {/* 
      // Content Section (Tabs and Details)
      <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: isWideScreen ? 'row' : 'column', overflow: 'hidden' }}>
        // Tabs
        <Tabs
          // ... existing props ...
          orientation={isWideScreen ? 'vertical' : 'horizontal'}
          variant="scrollable"
          // value={activeTab} // Commented out state
          // onChange={handleTabChange} // Commented out handler
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
          {/* Ensure availableTabs is not empty before mapping - availableTabs is commented out */}
          {/* 
          {availableTabs.length > 0 ? availableTabs.map((tab) => (
            <Tab key={tab.value} label={tab.label} value={tab.value} disabled={tab.disabled} />
          )) : (
             <Tab key="loading" label="Loading..." value="loading" disabled />
          )}
          */}
             <Tab key="loading" label="Loading..." value="loading" disabled /> { /* Placeholder while tabs are disabled */}
        </Tabs>

        // Tab Content
        <Box sx={{ p: isWideScreen ? 3 : 2, flexGrow: 1, overflowY: 'auto', backgroundColor: isDarkMode ? '#222' : '#fff' }}>
          {/* Ensure availableTabs is not empty before mapping - availableTabs is commented out */}
          {/*
          {availableTabs.length > 0 ? availableTabs.map((tab) => (
            <div key={tab.value} role="tabpanel" hidden={activeTab !== tab.value}>
              {activeTab === tab.value && tab.content()}
            </div>
          )) : (
             <CircularProgress sx={{display: 'block', margin: '2rem auto'}}/>
          )}
          */}
             <CircularProgress sx={{display: 'block', margin: '2rem auto'}}/> { /* Placeholder while tabs are disabled */}
        </Box>
      </Box>
      */
      {/* --- END DEBUG --- */}
    </Paper>
  );
});

export default WordDetails;