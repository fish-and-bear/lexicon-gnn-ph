import React, { useCallback, useState, useEffect, useRef } from 'react';
import { RawDefinition, WordInfo, WordForm, WordTemplate, Affixation, Credit, BasicWord, EtymologyTree, WordSuggestion, Example, WordNetwork, NetworkLink, NetworkNode } from '../types'; // Added EtymologyTree and WordSuggestion
// import { convertToBaybayin } from '../api/wordApi';
import './WordDetails.css';
// Import color utility functions needed
import { getNodeColor, getTextColorForBackground } from '../utils/colorUtils'; 
// *** ADD IMPORT for new language utility ***
import { getLanguageDisplayName } from '../utils/languageUtils';
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
  semanticNetworkData: WordNetwork | null; // Add prop for network data
  etymologyTree: EtymologyTree | null; // Use EtymologyTree type
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  onFetchEtymology: (wordId: number) => Promise<EtymologyTree | null>; 
  onWordClick: (word: string | WordSuggestion | BasicWord | null) => void; // Use WordSuggestion/BasicWord
  isMobile: boolean; 
  isLoading: boolean; // Add isLoading prop for details
}

// Helper function to format relation type names
const formatRelationType = (type: string): string => {
  // Special case for specific types
  if (type === 'kaugnay') return 'Kaugnay';
  if (type === 'kasalungat') return 'Kasalungat';
  if (type === 'kahulugan') return 'Kahulugan';
  if (type === 'has_translation' || type === 'translation_of') return 'Translation';
  // Consolidate usage/info/other/associated into 'Related'
  if (['usage', 'see_also', 'compare_with', 'associated', 'other'].includes(type.toLowerCase())) return 'Related';
  
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
  
  // Add translation colors (using related color)
  has_translation: "#48cae4", // Use related color
  translation_of: "#48cae4",  // Use related color
  
  // Origin group - Reds and oranges
  root_of: "#e63946",       // Add color for root_of (same as root)
  derived_from: "#f77f00",  // Color for derived_from
  etymology: "#d00000", // Dark red
  cognate: "#ff5c39",   // Color for cognate
  
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
  usage: "#fcbf49",     // Gold - Keep color definition but won't be used directly
  
  // Specific Filipino relations - Oranges and pinks
  kaugnay: "#fb8500",   // Orange
  salita: "#E91E63",    // Pink
  kahulugan: "#c9184a", // Dark pink
  kasalungat: "#e63946", // Red
  
  // Fallback
  // associated: "#adb5bd", // REMOVE - Mapped to related
  see_also: "#fcbf49", // REMOVE - Mapped to related
  compare_with: "#fcbf49", // REMOVE - Mapped to related
  // other: "#6c757d"      // REMOVE - Mapped to related
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
const StyledAccordion = styled(Accordion)(({ theme }: { theme: Theme }) => ({
  border: 'none',
  boxShadow: 'none',
  // Use elevated background for the whole accordion in dark mode
  backgroundColor: theme.palette.mode === 'dark' ? 'var(--card-bg-color-elevated)' : 'transparent',
  borderRadius: theme.shape.borderRadius, // Add some rounding
  marginBottom: theme.spacing(1), // Add some space between accordions
  '&:not(:last-child)': {
    // borderBottom: `1px solid ${theme.palette.mode === 'dark' ? 'rgba(255, 255, 255, 0.08)' : theme.palette.divider}`, // Remove individual bottom border
    borderBottom: 'none',
  },
  '&::before': {
    display: 'none',
  },
  // Ensure consistent background even when expanded
  '&.Mui-expanded': {
    margin: '0 0 8px 0', // Keep consistent margin
    backgroundColor: theme.palette.mode === 'dark' ? 'var(--card-bg-color-elevated)' : 'transparent',
  }
}));

const StyledAccordionSummary = styled(AccordionSummary)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(0, 2), // Increase horizontal padding
  minHeight: 48,
  // Make summary slightly lighter/different than details in dark mode
  backgroundColor:
    theme.palette.mode === 'dark'
      ? alpha(theme.palette.background.paper, 0.6) // Use paper bg with alpha
      : 'rgba(0, 0, 0, .02)',
  borderRadius: 'inherit', // Inherit border radius
  borderBottomLeftRadius: 0,
  borderBottomRightRadius: 0,
  '&:hover': {
     backgroundColor:
        theme.palette.mode === 'dark'
          ? alpha(theme.palette.background.paper, 0.8)
          : 'rgba(0, 0, 0, .03)',
  },
  '& .MuiAccordionSummary-content': {
    margin: theme.spacing(1.5, 0),
    alignItems: 'center',
  },
  '& .MuiAccordionSummary-expandIconWrapper.Mui-expanded': {
    transform: 'rotate(180deg)',
  },
  // Remove bottom radius when expanded to connect with details
  '&.Mui-expanded': {
    borderBottomLeftRadius: 0,
    borderBottomRightRadius: 0,
  }
}));

const StyledAccordionDetails = styled(AccordionDetails)(({ theme }: { theme: Theme }) => ({
  padding: theme.spacing(2, 2, 2, 2),
  borderTop: 'none',
  // Details bg should match the overall accordion bg in dark mode
  backgroundColor: theme.palette.mode === 'dark' ? 'var(--card-bg-color-elevated)' : 'transparent',
  borderBottomLeftRadius: 'inherit',
  borderBottomRightRadius: 'inherit'
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
    semanticNetworkData,
    etymologyTree,
    isLoadingEtymology,
    etymologyError,
    onFetchEtymology,
    onWordClick,
    isMobile,
    isLoading
  },
  ref
): React.ReactElement | null => { // Specify return type here
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const [activeTab, setActiveTab] = useState<string>('definitions');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  
  // Import NetworkLink and NetworkNode for typing
  // (Assuming they are exported from ../types)
  type NetworkLink = import('../types').NetworkLink;
  type NetworkNode = import('../types').NetworkNode;
  
  // Effect to setup audio element
  useEffect(() => {
    // Always reset audio playing state when word changes
    setIsAudioPlaying(false);
    setAudioElement(null);

    // Safely check for audio pronunciations
    if (!wordData || !wordData.pronunciations || !Array.isArray(wordData.pronunciations)) {
      return; // Exit early if no valid pronunciation data
    }

    // Find audio pronunciation with careful null checking
    const audioPronunciation = wordData.pronunciations.find(
      p => p && p.type === 'audio' && typeof p.value === 'string' && p.value.trim() !== ''
    );

    // Only proceed if we have a valid audio URL
    if (!audioPronunciation || !audioPronunciation.value) {
      return; // No valid audio pronunciation found
    }

    try {
      const audio = new Audio(audioPronunciation.value);
      
      if (!audio) {
        console.error("Failed to create audio element");
        return;
      }
      
      const onEnded = () => setIsAudioPlaying(false);
      audio.addEventListener('ended', onEnded);
      setAudioElement(audio);

      return () => {
        try {
          if (audio) {
            audio.pause();
            audio.removeEventListener('ended', onEnded);
          }
        } catch (cleanupError) {
          console.error("Error cleaning up audio element:", cleanupError);
        }
      };
    } catch (error) {
      console.error("Error creating audio element:", error);
      setAudioElement(null); // Ensure state is cleared on error
    }
  }, [wordData]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: string) => { // Restored event parameter
    setActiveTab(newValue);
  };

  const playAudio = useCallback(() => {
    if (!audioElement) {
      console.warn("Attempted to play audio, but no audio element available");
      return;
    }
    
    try {
      if (isAudioPlaying) {
        audioElement.pause();
        audioElement.currentTime = 0;
        setIsAudioPlaying(false);
      } else {
        audioElement.play()
          .then(() => setIsAudioPlaying(true))
          .catch(err => {
            console.error("Audio play failed:", err);
            setIsAudioPlaying(false); // Reset state on error
          });
      }
    } catch (error) {
      console.error("Error controlling audio playback:", error);
      setIsAudioPlaying(false);
    }
  }, [audioElement, isAudioPlaying]);

  // --- Rendering Sections ---

  const renderHeader = () => {
    // Find ALL IPA pronunciations
    const ipaPronunciations = wordData.pronunciations?.filter(p => p.type?.toLowerCase() === 'ipa') || [];
    const audioPronunciation = wordData.pronunciations?.find(p => p.type === 'audio' && p.value);

    // !!! ADDED DEBUG: Log pronunciation data
    // console.log('WordDetails Pronunciations:', wordData.pronunciations);
    // !!! END DEBUG

    return (
      <Paper 
        elevation={0} 
        className="word-details-header" 
        sx={{ 
          padding: theme.spacing(isMobile ? 1.5 : 2, 2), // Responsive padding
          marginBottom: theme.spacing(1),
          // backgroundColor: theme.palette.mode === 'dark' ? 'var(--card-bg-color)' : '#f8f9fa', // Standard card background
          backgroundColor: 'var(--card-bg-color)', // Use variable for both modes
          borderBottom: `1px solid ${theme.palette.divider}`,
          borderRadius: '4px 4px 0 0' // Rounded top corners
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <Box sx={{ flexGrow: 1 }}>
            <Typography 
              variant={isMobile ? "h5" : "h4"} 
              component="h1" 
              className="word-lemma"
              sx={{ 
                fontWeight: 600, 
                color: 'var(--text-color-highlight)', // Use highlight color
                mb: 0.5, 
                letterSpacing: '0.5px' 
              }}
            >
              {wordData.lemma}
            </Typography>
            
            {/* IPA Pronunciation(s) Display */}
            {ipaPronunciations.length > 0 && (
              // Changed Stack direction to 'row' and added wrapping + gap
              <Stack direction="row" spacing={1} sx={{ mb: 1, flexWrap: 'wrap', gap: 0.5 }}>
                {ipaPronunciations.map((pron, index) => (
                  // Removed the outer Box, relying on Stack spacing
                  <React.Fragment key={index}>
                    <Typography variant="body1" component="span" className="pronunciation-ipa" sx={{ color: 'text.secondary', fontStyle: 'italic', display: 'inline-block' }}>
                       /{pron.value}/
                    </Typography>
                    {pron.tags && pron.tags.length > 0 && (
                      <Chip 
                        label={pron.tags.join(', ')} 
                        size="small" 
                        variant="outlined" 
                        // Adjust chip styling for inline display
                        sx={{ 
                          height: 'auto', 
                          verticalAlign: 'middle', // Align chip nicely with text
                          ml: 0.5, // Add some space before the chip
                          '& .MuiChip-label': { py: '1px', fontSize: '0.7rem' } 
                        }} 
                      />
                    )}
                  </React.Fragment>
                ))}
              </Stack>
            )}

            {/* Language Code Chip */}
            {wordData.language_code && (
              <Chip 
                // *** USE THE NEW FUNCTION HERE ***
                label={getLanguageDisplayName(wordData.language_code)} 
                size="small" 
                variant="outlined" 
                sx={{ fontWeight: 500, mb: 1, mr: 1 }} 
                // Keep tooltip showing raw code for debugging/info
                title={`Raw code: ${wordData.language_code}`}
              />
            )}
            
            {/* Baybayin Form (if available) - Enhanced Styling */}
            {wordData.has_baybayin && wordData.baybayin_form && (
               <Typography 
                 variant="h5" 
                 component="div" 
                 className="baybayin-text" 
                 lang="tl-Tglg" // Specify language tag for Baybayin
                 sx={{ 
                   // Use the specific Baybayin font class defined in CSS
                   // Apply margin top for spacing, adjust font size/color
                   mt: 1.5, // Add space above
                   mb: 1,   // Space below
                   color: 'var(--accent-color)', // Use accent color for visibility
                   fontSize: isMobile ? '1.8rem' : '2.5rem', // *** Slightly smaller on mobile ***
                   lineHeight: 1.1,
                   textAlign: 'left', // Align with lemma
                   // Ensure background isn't interfering if set in CSS
                   backgroundColor: 'transparent' 
                 }}
               >
                 {wordData.baybayin_form}
               </Typography>
            )}
          </Box>

          {/* Audio Player Button */}
          {audioPronunciation && (
            <IconButton 
              onClick={playAudio} 
              size="small" 
              aria-label={isAudioPlaying ? "Stop audio" : "Play audio pronunciation"}
              title={isAudioPlaying ? "Stop audio" : "Play audio pronunciation"}
              sx={{
                ml: 1, // Margin left
                mt: -0.5, // Adjust vertical position
                color: 'primary.main',
                border: `1px solid ${alpha(theme.palette.primary.main, 0.3)}`,
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.15),
                }
              }}
            >
              {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
            </IconButton>
          )}
        </Box>
      </Paper>
    );
  };

  const renderDefinitionsTab = () => {
    if (!wordData.definitions || wordData.definitions.length === 0) {
      return <Alert severity="info">No definitions available for this word.</Alert>;
    }

    // Group by Combined POS (English + Tagalog if different) then Source
    const definitionsByCombinedPosThenSource: { [combinedPosKey: string]: { [source: string]: RawDefinition[] } } = {};
    wordData.definitions.forEach((def: RawDefinition) => {
      const englishPos = def.standardized_pos?.name_en || def.standardized_pos?.code || def.original_pos || 'Other';
      const tagalogPos = def.standardized_pos?.name_tl;
      // Create a combined key for grouping
      const combinedPosKey = tagalogPos && tagalogPos !== englishPos ? `${englishPos} (${tagalogPos})` : englishPos;
      
      const sourceKey = def.sources || 'Unknown Source'; 
      
      if (!definitionsByCombinedPosThenSource[combinedPosKey]) {
        definitionsByCombinedPosThenSource[combinedPosKey] = {};
      }
      if (!definitionsByCombinedPosThenSource[combinedPosKey][sourceKey]) {
        definitionsByCombinedPosThenSource[combinedPosKey][sourceKey] = [];
      }
      definitionsByCombinedPosThenSource[combinedPosKey][sourceKey].push(def);
    });

    return (
      <Box sx={{ pt: isMobile ? 0.5 : 1, width: '100%', maxWidth: '100%', pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}> 
        {/* Iterate using the new combined key */}
        {Object.entries(definitionsByCombinedPosThenSource).map(([combinedPosKey, defsBySource]) => {
          return (
          <Box key={combinedPosKey} sx={{ mb: isMobile ? 1.5 : 3, width: '100%', maxWidth: '100%' }}>
            {/* Display the combined POS Key directly in the header */}
            <Typography 
              variant="subtitle1" 
              component="h3" 
              sx={{ 
                color: graphColors.main, 
                fontWeight: 600,
                pb: isMobile ? 0.5 : 1,
                fontSize: isMobile ? '0.85rem' : '1rem', // *** Reduced mobile font size ***
                borderBottom: `1px solid ${alpha(theme.palette.divider, 0.6)}`,
                mb: isMobile ? 1 : 1.5,
                width: '100%',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
            >
              {combinedPosKey} {/* Display the combined key */}
            </Typography>
            
            {Object.entries(defsBySource).map(([sourceName, defs]) => {
              // Filtering logic remains the same
              const seenBaseDefinitions = new Set<string>();
              const filteredDefs = defs.filter((def) => {
                const baseText = stripLeadingNumber(def.definition_text);
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
                  {filteredDefs.map((def: RawDefinition, index: number) => {
                    const isLastDefinitionForSource = index === filteredDefs.length - 1;

                    return (
                    <ListItem 
                      key={def.id || index} 
                      alignItems="flex-start" 
                      sx={{ /* Existing ListItem sx */ 
                        flexDirection: 'column', 
                        gap: 0.5, 
                        py: isMobile ? 1 : 1.5, 
                        pl: 0, 
                        pr: 0, 
                        position: 'relative', 
                        pb: isLastDefinitionForSource && sourceName !== 'Unknown Source' ? (isMobile ? 2.5 : 3) : (isMobile ? 1 : 1.5), 
                      }}
                    >
                      {/* Definition text - REMOVE secondary TL display & strip trailing digits */}
                      <ListItemText
                        primary={
                          <Typography component="div" variant="body1" fontWeight={500} fontSize={isMobile ? '0.85rem' : '1rem'} pl={0.5} lineHeight={isMobile ? 1.5 : 1.6}>
                            {def.definition_text.replace(/\d+$/, '').trim()} {/* Strip trailing digits */}
                          </Typography>
                        }
                        // secondary={...} // REMOVED secondary prop
                      />
                      
                      {/* Examples (Keep as is) */}
                      {def.examples && def.examples.length > 0 && (
                        <Box sx={{ pl: 1, mb: 1 }}> 
                          {def.examples!.map((example: Example, exIndex: number) => ( 
                            <Box 
                              key={exIndex} 
                              className="definition-example-text" 
                              sx={{ /* Existing example sx */ 
                                mb: exIndex < def.examples!.length - 1 ? 1 : 0, 
                                p: 1.5, 
                                bgcolor: theme => alpha(theme.palette.grey[500], 0.05), 
                                borderRadius: 1, 
                                borderLeft: `3px solid ${alpha(theme.palette.primary.main, 0.3)}`, 
                              }}
                            >
                              <Typography component="div" sx={{ fontStyle: 'normal', color: 'text.primary', mb: 0.5 }}>
                                {example.example_text}
                              </Typography>
                              {example.translation && (
                                <Typography component="div" sx={{ fontStyle: 'italic', color: 'text.secondary', fontSize: '0.85em' }}>
                                   - {example.translation}
                                </Typography>
                              )}
                              {example.example_metadata?.romanization && 
                                <Typography component="div" color="text.disabled" sx={{ fontStyle: 'italic', fontSize: '0.8em', mt: 0.5 }}>
                                  ({example.example_metadata.romanization})
                                </Typography>
                              }
                            </Box>
                          ))}
                        </Box>
                      )}
                      
                      {/* Usage notes (Keep as is) */}
                      {def.usage_notes && (
                        <Box sx={{ pl: 1, mb: 1 }}>
                          <Typography 
                            component="div" 
                            sx={{ fontWeight: 500, mb: 0.5, fontSize: isMobile ? '0.75rem' : '0.8rem' }}
                          >
                            Usage Notes:
                          </Typography>
                          <Typography 
                              component="div" 
                              sx={{ color: 'text.secondary', fontSize: isMobile ? '0.8rem' : '0.875rem' }}
                            >
                              {def.usage_notes}
                            </Typography>
                        </Box>
                      )}
                      
                      {/* Definition tags (Keep as is) */}
                      {def.tags && def.tags.trim().length > 0 && (
                        <Stack 
                          direction="row" 
                          spacing={0.5} 
                          useFlexGap 
                          flexWrap="wrap" 
                          sx={{ mt: 0.5 }}
                        >
                          {def.tags.split(',').map(tag => tag.trim()).filter(tag => tag).map((tag: string, tagIndex: number) => (
                            <Chip
                              key={tagIndex}
                              label={tag}
                              size="small"
                              variant="outlined" // Use outlined variant
                              sx={{
                                fontSize: isMobile ? '0.65rem' : '0.7rem', 
                                height: 'auto',
                                padding: theme.spacing(0.1, 0), // Adjust padding
                                // Use neutral theme colors
                                color: 'text.secondary',
                                borderColor: 'action.disabledBackground',
                                bgcolor: 'transparent', // Ensure no background override
                                '& .MuiChip-label': { 
                                  px: 0.75, 
                                  py: 0.25 // Adjust label padding
                                }
                              }}
                            />
                          ))}
                        </Stack>
                      )}

                      {/* Source Chip (Keep as is) */}
                      {isLastDefinitionForSource && sourceName !== 'Unknown Source' && (
                        <Chip
                          label={`${sourceName}`}
                          size="small"
                          variant="outlined"
                          sx={{ /* Existing source chip sx */
                            position: 'absolute', 
                            bottom: theme.spacing(isMobile ? 0.25 : 0.5),
                            right: theme.spacing(isMobile ? 0.25 : 0.5),
                            fontSize: isMobile ? '0.65rem' : '0.7rem', 
                            height: 'auto',
                            padding: theme.spacing(0.25, 0),
                            borderColor: alpha(graphColors.related, 0.4),
                            color: alpha(graphColors.related, 0.9),
                            bgcolor: alpha(graphColors.related, 0.04),
                            maxWidth: 'calc(100% - 16px)', 
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            zIndex: 1, 
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
    
    // Use the semanticNetworkData prop
    const semanticNetworkNodes = semanticNetworkData?.nodes || [];
    const semanticNetworkLinks = semanticNetworkData?.links || [];
    
    // Determine if semantic network fallback should be used
    const useSemanticNetwork = 
      (incoming_relations.length === 0 && outgoing_relations.length === 0) && 
      semanticNetworkLinks.length > 0;
    
    // Helper function to create relation objects from semantic network data
    function createRelationsFromNetwork(): any[] { // Return type can be refined if needed
      console.log("Creating relations from semantic network as fallback");
      const mainWord = wordData.lemma;
      const relations: any[] = [];
      
      if (!semanticNetworkLinks || !Array.isArray(semanticNetworkLinks) || !semanticNetworkNodes || !Array.isArray(semanticNetworkNodes)) {
        console.warn("Invalid semantic network data structure");
        return relations;
      }
      
      semanticNetworkLinks.forEach((link: NetworkLink) => { // Type link
        if (!link) return; // Skip invalid links
        
        // Safely determine target ID with type checking
        let targetId: number | string | null = null;
        
        try {
          if (typeof link.target === 'object' && link.target && link.target.id) {
            targetId = link.target.id;
          } else if (typeof link.target === 'number' || typeof link.target === 'string') {
            targetId = link.target;
          }
          
          if (targetId === null) {
            console.warn("Could not determine target ID from link:", link);
            return; // Skip this link
          }
          
          // Find the connected node
          const targetNode = semanticNetworkNodes.find(
            (n: NetworkNode) => n && n.id === targetId
          );
          
          if (!targetNode) {
            console.warn(`Target node not found for ID: ${targetId}`);
            return;
          }
          
          // Calculate degree/distance (default to 1 for direct connections)
          // Check if link object itself has degree/distance, otherwise default to 1
          const degree = (link as any).distance || (link as any).degree || 1; 
          
          // Ensure targetNode has the required properties
          const nodeLabel = targetNode.label || targetNode.word || String(targetNode.id);
          
          // Create a relation object mimicking standard Relation structure + degree
          relations.push({
            id: `semantic-${targetNode.id}`,
            relation_type: link.type || 'related',
            degree: degree, // Include degree
            // Create a wordObj similar to target_word/source_word in Relation
            wordObj: {
              id: targetNode.id,
              lemma: nodeLabel,
              has_baybayin: targetNode.has_baybayin || false,
              baybayin_form: targetNode.baybayin_form || null
              // Add other BasicWord fields if needed (e.g., language_code, gloss, pos)
            }
          });
        } catch (error) {
          console.error("Error processing semantic network link:", error, link);
        }
      });
      
      console.log(`Created ${relations.length} fallback relations from semantic network`);
      return relations;
    }
    
    // Create relations from semantic network if needed
    const semanticRelations = useSemanticNetwork ? createRelationsFromNetwork() : [];
    console.log(`[REL TAB] Using semantic network fallback: ${useSemanticNetwork}`); // Log fallback status
    if (useSemanticNetwork) {
      console.log("[REL TAB] Fallback semanticRelations:", semanticRelations);
    }
    
    // Check if there are any standard relations or fallback relations
    const hasStandardRelations = (incoming_relations.length > 0 || outgoing_relations.length > 0);
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
                        primary={
                          <Typography component="div" variant="body2" sx={{ cursor: affix.affixed_word ? 'pointer' : 'default', textDecoration: affix.affixed_word ? 'underline' : 'none', color: affix.affixed_word ? theme.palette.primary.main : 'inherit', mr: 1 }}>
                            {affix.affixed_word?.lemma || 'Unknown'}
                          </Typography>
                        }
                        secondary={
                          <Typography component="div" variant="caption" color="text.secondary">
                            (as {affix.affix_type})
                          </Typography>
                        }
                        primaryTypographyProps={{ component: 'span' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
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
                        primary={
                          <Typography component="div" variant="body2" sx={{ cursor: affix.root_word ? 'pointer' : 'default', textDecoration: affix.root_word ? 'underline' : 'none', color: affix.root_word ? theme.palette.primary.main : 'inherit', mr: 1 }}>
                            {affix.root_word?.lemma || 'Unknown'}
                          </Typography>
                        }
                        secondary={
                          <Typography component="div" variant="caption" color="text.secondary">
                            (using {affix.affix_type})
                          </Typography>
                        }
                        primaryTypographyProps={{ component: 'span' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
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
    const relationsWithDegree = allRelations.map(rel => {
      // Determine the correct related word object based on direction
      let relatedWordObj = null;
      if (rel.target_word) { // Check if it's an outgoing relation structure from wordData
          relatedWordObj = rel.target_word;
      } else if (rel.source_word) { // Check if it's an incoming relation structure from wordData
          relatedWordObj = rel.source_word;
      } else { // Fallback for semantic network structure or other formats
          relatedWordObj = rel.wordObj;
      }

      return {
        ...rel,
        wordObj: relatedWordObj, // Assign the determined related word object
        degree: rel.degree || 1 // Default to 1 (direct connection) if not specified
      };
    });
    
    // Filter out self-references (where the related word is the same as the main word)
    console.log("[REL TAB] Relations before self-ref filter (relationsWithDegree):", relationsWithDegree); // <-- ADD LOG HERE
    const filteredRelations = relationsWithDegree.filter(rel => {
      return rel.wordObj?.lemma !== wordData.lemma;
    });
    console.log("[REL TAB] Filtered relations (after self-ref check):", filteredRelations); // Log filtered relations
    
    // Define relation categories and their types
    const relationCategories = [
      {
        name: "Origin",
        types: ["root", "root_of", "derived_from", "etymology", "cognate"], 
        color: relationColors.root 
      },
      {
        name: "Derivation", 
        types: ["derived", "derivative", "sahod", "isahod", "affix"], 
        color: relationColors.derived 
      },
      {
        name: "Meaning",
        types: [
          "synonym", "antonym", "related", "similar", "kaugnay", "kahulugan", "kasalungat", 
          "has_translation", "translation_of", 
          // Add previously Info/Other types
          "usage", "see_also", "compare_with", "associated", "other" 
        ], 
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
      }
      // REMOVE Info and Other categories
      // {
      //   name: "Info", 
      //   types: ["usage", "see_also", "compare_with", "associated"], 
      //   color: relationColors.usage 
      // },
      // {
      //   name: "Other",
      //   types: ["other"],
      //   color: relationColors.other
      // }
    ];
    
    // COMPLETE RESTRUCTURE: Deduplicate first, then categorize once
    
    // Step 1: Group by word lemma, collecting all relation types for each word
    const wordMap = new Map<string, any>();
    
    filteredRelations.forEach(relation => {
      // Use lemma, label, or word as the key for broader compatibility
      const wordKey = relation.wordObj?.lemma || relation.wordObj?.label || relation.wordObj?.word;
      if (!wordKey) {
        console.warn("Skipping relation due to missing lemma/label/word in wordObj:", relation.wordObj);
        return; // Skip if no usable key
      }
      
      if (!wordMap.has(wordKey)) {
        // First time seeing this word, create a new entry
        const newRelation = { 
          ...relation,
          relationTypes: [relation.relation_type],
          originalRelation: relation
        };
        wordMap.set(wordKey, newRelation);
      } else {
        // Word exists, just add the relation type if new
        const existingRelation = wordMap.get(wordKey);
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
    categorizedWords["Other"] = []; // *** ADD THIS LINE TO INITIALIZE "Other" ***
    
    // Debug the data we're working with
    console.log("Deduplicating relations. Unique words count:", wordMap.size);
    console.log("Word lemmas:", Array.from(wordMap.keys()));
    console.log("[REL TAB] Word Map content before count:", wordMap); // Log wordMap content
    
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
      <Box sx={{ pt: theme.spacing(1), width: '100%', maxWidth: '100%', overflow: 'hidden', pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}>
        {/* ADDED Call to render Affixations */} 
        {renderAffixations()}

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
                                  onClick={() => {
                                    if (typeof onWordClick === 'function' && (wordObj.lemma || wordObj.word)) {
                                      onWordClick(wordObj.lemma || wordObj.word);
                                    } else {
                                      console.warn("Unable to navigate: onWordClick is not a function or word is undefined");
                                    }
                                  }}
                                  variant="outlined"
                                  sx={{
                                    // Base styles - Subtle background, colored text/border
                                    fontSize: '0.75rem',
                                    height: 'auto',
                                    padding: theme.spacing(0.25, 0),
                                    my: 0.5,
                                    maxWidth: '100%',
                                    bgcolor: alpha(relColor, 0.08), // Subtle background always
                                    color: relColor, // Text color always the relation color
                                    borderColor: alpha(relColor, 0.5), // Border color always related to relation color
                                    borderWidth: '1px',
                                    borderStyle: 'solid',
                                    
                                    // Dark Mode Adjustments (if needed, but base might work)
                                    ...(theme.palette.mode === 'dark' && {
                                      // Keep subtle background, ensure text is bright enough if needed
                                      color: alpha(relColor, 0.9), // Slightly brighter text in dark maybe?
                                      borderColor: alpha(relColor, 0.6), // Slightly brighter border
                                      bgcolor: alpha(relColor, 0.15), // Slightly more opaque bg in dark?
                                    }),
                                    
                                    '& .MuiChip-label': { 
                                      px: 1, 
                                      py: 0.25, 
                                      fontWeight: 500,
                                      width: '100%',
                                      maxWidth: '100%',
                                      overflow: 'hidden'
                                    },
                                    // Hover styles - Subtle brightness/alpha changes
                                    '&:hover': {
                                      bgcolor: alpha(relColor, 0.15), // Slightly darken bg on hover
                                      borderColor: alpha(relColor, 0.7),
                                      ...(theme.palette.mode === 'dark' && {
                                        bgcolor: alpha(relColor, 0.25),
                                        borderColor: alpha(relColor, 0.8),
                                        color: relColor, // Ensure full color text on hover in dark
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
        {import.meta.env.DEV && typeof wordData.completeness_score === 'number' && (
          <Box sx={{ mt: theme.spacing(3), pt: theme.spacing(2), borderTop: `1px solid ${theme.palette.divider}` }}>
            <Typography variant="subtitle2" sx={{ mb: theme.spacing(1) }}>
              Completeness Score
            </Typography>
            {/* Display the score directly */}
            <Chip
                label={`${(wordData.completeness_score * 100).toFixed(0)}%`}
                color="success"
                variant="filled"
                size="small"
                sx={{ justifyContent: 'flex-start' }}
            />
            {/* Remove iteration over old data_completeness object */}
            {/* <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 1 }}>
              {Object.entries(wordData.data_completeness).map(([key, value]) => (...))}
            </Box> */}
          </Box>
        )} {/* ADD THIS LINE to close the conditional block */}
      </Box>
    );
  };

  const renderFormsAndTemplatesTab = () => {
    const hasForms = wordData.forms && wordData.forms.length > 0;
    const hasTemplates = wordData.templates && wordData.templates.length > 0;

    if (!hasForms && !hasTemplates) {
      return <Typography sx={{ p: 2, fontStyle: 'italic', color: 'text.secondary' }}>No forms or templates available for this word.</Typography>;
    }

    // Helper to render tags object as chips
    const renderTags = (tags: Record<string, any> | null | undefined, isCanonicalForm: boolean = false) => {
      if (!tags || Object.keys(tags).length === 0) return null;

      // Filter out redundant/unnecessary tags
      const filteredTags = Object.entries(tags)
        .filter(([key, value]) => {
          // Always skip 'is_canonical' key, regardless of value
          if (key === 'is_canonical') return false;
          
          // Skip 'tags: canonical' only if the form is canonical
          if (isCanonicalForm && key === 'tags' && String(value).toLowerCase() === 'canonical') {
              return false;
          }
          
          // Keep all other tags
          return true;
        });
        
      if (filteredTags.length === 0) return null; // Return null if no tags remain after filtering

      return (
        <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mt: 0.5 }}>
          {filteredTags.map(([key, value]) => (
                            <Chip 
              key={key} 
              label={`${key}: ${String(value)}`} 
                              size="small" 
              variant="outlined" 
                              sx={{ 
                fontSize: '0.75rem',
                                height: 'auto', 
                lineHeight: 1.2,
                padding: '1px 4px',
                borderColor: 'action.disabledBackground'
                              }} 
                            />
          ))}
                        </Stack>
      );
    };

    return (
      <Box sx={{ p: isMobile ? 1 : 2, pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}>
        {/* Word Forms Section */}
        {hasForms && (
          <Box mb={hasTemplates ? 3 : 0}>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', mb: 1.5 }}>
              Word Forms
            </Typography>
            <Stack spacing={2}>
              {wordData.forms?.map((form, index) => (
                <Paper 
                  key={form.id || `form-${index}`}
                  variant="outlined"
                        sx={{ 
                    p: 1.5,
                    borderRadius: '8px',
                    borderColor: 'rgba(0, 0, 0, 0.08)',
                    bgcolor: isDarkMode ? alpha(theme.palette.grey[900], 0.5) : alpha(theme.palette.grey[50], 0.7),
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                    <Typography variant="body1" sx={{ fontWeight: 500, flexGrow: 1 }}>
                      {form.form}
                      </Typography>
                    {form.is_canonical && (
                          <Chip 
                        label="Canonical"
                        color="primary" // Keep primary color association
                        variant="outlined" // Change to outlined
                            size="small" 
                            sx={{ 
                              height: 'auto', 
                          fontSize: '0.75rem',
                          fontWeight: 600,
                          // bgcolor: isDarkMode ? alpha(graphColors.main, 0.8) : alpha(graphColors.main, 0.15), // Remove bgcolor
                          // color: isDarkMode ? '#fff' : graphColors.main, // Let outlined handle color
                          // border: `1px solid ${alpha(graphColors.main, 0.4)}` // Let outlined handle border
                          borderColor: alpha(graphColors.main, 0.6), // Slightly adjust border alpha if needed
                          color: graphColors.main // Ensure text color is primary
                          }} 
                        />
                      )}
                    </Box>
                  {/* Pass is_canonical flag to renderTags, handling null */} 
                  {renderTags(form.tags, form.is_canonical ?? false)}
                  {form.sources && (
                    <Typography variant="caption" display="block" sx={{ mt: 1, color: 'text.secondary', fontStyle: 'italic' }}>
                      Source: {form.sources}
                    </Typography>
                  )}
                </Paper>
                ))}
            </Stack>
              </Box>
        )}

        {/* Word Templates Section (existing code) */}
        {hasTemplates && (
          <Box>
            <Typography variant="h6" gutterBottom sx={{ fontSize: '1.1rem', mb: 1.5 }}>
              Word Templates
            </Typography>
            {/* {wordData.templates?.map((template, index) => ( // <-- Re-comment this line and the closing parenthesis/brace
              <Accordion key={template.id || `template-${index}`} defaultExpanded={index < 2}> 
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography sx={{ fontWeight: 500 }}>{template?.template_name || 'Unnamed Template'}</Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ bgcolor: 'action.hover' }}>
                  {template.expansion && (
                    <Box mb={1}>
                      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Expansion:</Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                        {template.expansion}
                      </Typography>
                    </Box>
                  )}
                  {template.args && Object.keys(template.args).length > 0 && (
                    <Box mb={1}>
                      <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Args:</Typography>
                      <Paper variant="outlined" sx={{ p: 1, bgcolor: 'background.default' }}>
                        <pre style={{ margin: 0, fontSize: '0.8rem', whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                          {JSON.stringify(template.args, null, 2)}
                        </pre>
                      </Paper>
                    </Box>
                  )}
                  {template.sources && (
                     <Typography variant="caption" display="block" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
                      Source: {template.sources}
                    </Typography>
                  )}
                </AccordionDetails>
              </Accordion>
            ))} */}
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
        <Box sx={{ p: isMobile ? 1.5 : 2, pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}>
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
    
    // Helper function to render a single field (like Components, Languages, Sources)
    const renderEtymologyField = (label: string, items: string[], chipColor: string, clickable: boolean = false) => {
      if (!items || items.length === 0) return null;
      
      // Determine text color based on chip background for dark mode
      const chipTextColor = isDarkMode ? getTextColorForBackground(chipColor) : chipColor;

      return (
        <Box sx={{ mt: 1.5 }}>
                      <Typography 
            variant="subtitle2" 
                        sx={{ 
              mb: 0.75,
                              fontWeight: 500, 
              fontSize: isMobile ? '0.75rem' : '0.8rem',
              color: 'text.secondary'
                            }}
                          >
            {label}:
                          </Typography>
                          <Stack direction="row" spacing={isMobile ? 0.5 : 1} useFlexGap flexWrap="wrap">
            {items.map((item: string, i: number) => (
                              <Chip 
                                key={i}
                label={item}
                                size="small"
                clickable={clickable}
                onClick={clickable ? () => onWordClick(item) : undefined}
                                sx={{ 
                                  fontSize: isMobile ? '0.7rem' : '0.75rem',
                                  height: isMobile ? 20 : 24,
                  // Base styles (Light mode defaults)
                  bgcolor: alpha(chipColor, 0.08),
                  color: chipColor, 
                                  fontWeight: 500,
                  border: `1px solid ${alpha(chipColor, 0.4)}`,
                  
                  // Dark Mode Styles override base
                  ...(isDarkMode && {
                      bgcolor: alpha(chipColor, 0.15),
                      color: alpha(chipColor, 0.9),
                      borderColor: alpha(chipColor, 0.6),
                  }),
                  
                  // Hover styles (applied over base/dark)
                  ...(clickable && {
                                  '&:hover': {
                      // Light mode hover defaults
                      bgcolor: alpha(chipColor, 0.15), 
                      borderColor: alpha(chipColor, 0.6),
                      // Dark mode hover overrides
                      ...(isDarkMode && { 
                          bgcolor: alpha(chipColor, 0.25),
                          borderColor: alpha(chipColor, 0.8),
                          color: chipColor, // Keep full color text on dark hover
                      })
                    }
                  })
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
      );
    };

    // If there's etymology data in the word itself, display it regardless of tree
    if (hasWordEtymologies) {
      return (
        <Box sx={{ p: isMobile ? 1 : 2, pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}>
          <Stack spacing={isMobile ? 1.5 : 2}>
            {wordData.etymologies!.map((etym, index) => {
              const components = etym.normalized_components?.split(/\s+/).filter(c => c.trim()) || [];
              const languages = etym.language_codes?.split(',').map(l => l.trim()).filter(l => l) || [];
              const sources = etym.sources?.split(',').map(s => s.trim()).filter(s => s) || [];
              
              return (
                <Paper 
                  key={index} 
                  variant="outlined" 
                            sx={{ 
                    p: isMobile ? 1.5 : 2,
                    borderRadius: '8px',
                    // Use elevated card background in dark mode
                    borderColor: isDarkMode ? 'var(--card-border-color)' : theme.palette.divider, 
                    bgcolor: isDarkMode ? 'var(--card-bg-color-elevated)' : alpha(theme.palette.grey[50], 0.5),
                  }}
                >
                  {/* Title */}
                  <Typography 
                    variant="subtitle1" 
                                sx={{ 
                      fontWeight: 600, 
                      mb: 1.5, 
                      pb: 0.5, 
                      color: graphColors.main, 
                      borderBottom: `1px solid ${alpha(graphColors.main, 0.2)}`,
                      fontSize: isMobile ? '0.9rem' : '1rem' 
                    }}
                  >
                    Etymology {index + 1}
                  </Typography>

                  {/* Main text */}
                  {etym.etymology_text && (
                          <Typography 
                            component="div" 
                            sx={{ 
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        lineHeight: 1.6, 
                        fontSize: isMobile ? '0.85rem' : '0.9rem',
                        mb: (components.length > 0 || languages.length > 0 || sources.length > 0) ? 2 : 0 // Add margin if fields below exist
                            }}
                          >
                      {etym.etymology_text}
                        </Typography>
                  )}

                  {/* Fields: Components, Languages, Sources */}
                  {renderEtymologyField('Components', components, graphColors.derived, true)} 
                  {renderEtymologyField('Languages', languages, graphColors.variant)} 
                  {renderEtymologyField('Sources', sources, graphColors.related)} 
                  
                </Paper>
              );
            })}
                          </Stack>
          
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
      <Box sx={{ p: 0, pb: isMobile ? theme.spacing(16) : 0 }}> {/* Ensure mobile padding for etymology tree only view */}
        {renderEtymologyTreeVisualization()}
      </Box>
    );
  };
  
  // Helper function to render the etymology tree visualization
  const renderEtymologyTreeVisualization = () => {
    // Use links instead of edges
    if (!etymologyTree || !etymologyTree.nodes || !etymologyTree.links || 
        !Array.isArray(etymologyTree.nodes) || !Array.isArray(etymologyTree.links)) {
      return <Alert severity="info">Etymology tree data is not available or has an invalid structure.</Alert>;
    }
  
    // Remove local EtymologyNode/Edge types, use NetworkNode/Link from import
    // type EtymologyNode = { id: number; label: string; language?: string; [key: string]: any };
    // type EtymologyEdge = { source: number; target: number; [key: string]: any };
    type EtymologyTreeMap = { [id: number]: NetworkNode }; // Use NetworkNode
  
    const renderNode = (nodeId: number, nodes: EtymologyTreeMap, links: NetworkLink[], level = 0): React.ReactNode => { // Use NetworkLink
      // Safety check for valid nodeId
      if (nodeId === undefined || nodeId === null) {
        console.warn("Invalid nodeId provided to renderNode");
        return null;
      }
      
      const node = nodes[nodeId];
      if (!node) {
        console.warn(`Node with ID ${nodeId} not found in node map`);
        return null;
      }

      // Use links instead of edges - with careful null checking
      const childrenLinks = links.filter(link => {
        if (!link) return false;
        
        let sourceId: any;
        try {
          sourceId = typeof link.source === 'object' ? 
            (link.source && link.source.id ? link.source.id : null) : 
            link.source;
          
          return sourceId === nodeId;
        } catch (error) {
          console.error("Error filtering child links:", error);
          return false;
        }
      });
  
      return (
        <Box key={node.id} sx={{ ml: level * 2.5, mb: 1.5 }}>
          <Paper elevation={1} sx={{ p: 1.5, display: 'flex', alignItems: 'center', gap: 1.5, borderRadius: '8px', bgcolor: alpha(getNodeColor(node.language || 'associated'), 0.1) }}>
            <Chip 
              label={(node.language && typeof node.language === 'string') ? node.language.toUpperCase() : 'UNK'} 
              size="small" 
              sx={{ fontWeight: 600, bgcolor: getNodeColor(node.language || 'associated'), color: getTextColorForBackground(getNodeColor(node.language || 'associated')) }}
            />
            <Link
              component="button"
              onClick={() => {
                if (typeof onWordClick === 'function') {
                  onWordClick(`id:${node.id}`);
                } else {
                  console.warn("onWordClick is not a function");
                }
              }} 
              sx={{ 
                fontWeight: 500, 
                fontSize: '1rem', 
                cursor: 'pointer', 
                color: 'text.primary', 
                '&:hover': { color: 'primary.main' }
              }}
            >
              {node.label || String(node.id)}
            </Link>
          </Paper>
          {childrenLinks.length > 0 && (
            <Box sx={{ 
              mt: isMobile ? 0.5 : 1, 
              pl: isMobile ? 1.5 : 2, 
              // Refine borderLeft for visual connection
              borderLeft: `1px solid ${alpha(theme.palette.divider, isMobile ? 0.7 : 0.5)}` 
            }}>
              {/* Use links instead of edges */} 
              {childrenLinks.map((link, index) => {
                if (!link) return null;
                
                try {
                  const targetId = typeof link.target === 'object' ? 
                    (link.target && link.target.id ? link.target.id : null) : 
                    link.target;
                  
                  if (targetId === null || targetId === undefined) {
                    console.warn("Invalid target ID in link", link);
                    return null;
                  }
                  
                  return renderNode(targetId, nodes, links, level + 1);
                } catch (error) {
                  console.error("Error rendering child node:", error);
                  return null;
                }
              })}
            </Box>
          )}
        </Box>
      );
    };
    
    try {
      // Use links instead of edges, type edge as NetworkLink
      // Create a more resilient filtering approach
      const rootNodes = [];
      
      // Create a set of all target IDs for faster lookups
      const targetIds = new Set();
      
      // Collect all target IDs from links
      for (const link of etymologyTree.links) {
        if (!link) continue;
        
        try {
          const targetId = typeof link.target === 'object' ? 
            (link.target && link.target.id ? link.target.id : null) : 
            link.target;
          
          if (targetId !== null && targetId !== undefined) {
            targetIds.add(targetId);
          }
        } catch (error) {
          console.error("Error processing link target:", error);
        }
      }
      
      // Find nodes that aren't targets of any link
      for (const node of etymologyTree.nodes) {
        if (!node || node.id === undefined) continue;
        
        if (!targetIds.has(node.id)) {
          rootNodes.push(node);
        }
      }
      
      // Build node map for quick lookup
      const nodeMap: Record<number, NetworkNode> = {};
      for (const node of etymologyTree.nodes) {
        if (node && node.id !== undefined) {
          nodeMap[node.id] = node;
        }
      }
      
      if (rootNodes.length === 0) {
        return <Alert severity="info">Could not determine root nodes in the etymology tree.</Alert>;
      }
      
      return (
        <Box sx={{ fontFamily: 'system-ui, sans-serif' }}>
          {/* Use links instead of edges */} 
          {rootNodes.map(rootNode => rootNode ? renderNode(rootNode.id, nodeMap, etymologyTree.links!) : null)}
        </Box>
      );
    } catch (error) {
      console.error("Error rendering etymology tree:", error);
      return <Alert severity="error">Error rendering etymology tree: {String(error)}</Alert>;
    }
  };

  // *** START: Re-insert renderSourcesInfoTab ***
  const renderSourcesInfoTab = () => {
     const credits = wordData.credits || [];
     const sourceInfo = wordData.source_info || {};
     // Use completeness_score instead of data_completeness
     const score = wordData.completeness_score; 

     const hasCredits = credits.length > 0;
     const hasSourceInfo = Object.keys(sourceInfo).length > 0;
     // Check if score is a valid number
     const hasScore = typeof score === 'number' && !isNaN(score);
     const hasEntryInfo = wordData.created_at || wordData.updated_at;

     if (!hasCredits && !hasSourceInfo && !hasScore && !hasEntryInfo) {
       return <Alert severity="info" sx={{ m: 2 }}>No source, metadata, or entry information available.</Alert>;
     }

     // Helper to render JSON data nicely (Keep for other potential JSON fields)
     const renderJsonData = (title: string, data: Record<string, any>) => {
       if (!data || Object.keys(data).length === 0) return null; // Added null check for data
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
     
     // --- NEW: Helper to render Source Information specifically ---
     const renderStructuredSourceInfo = (info: Record<string, any>) => {
        if (!info || Object.keys(info).length === 0) return null;

        // const files = info.files && Array.isArray(info.files) ? info.files : []; // No longer rendering files list
        const lastUpdated = info.last_updated && typeof info.last_updated === 'object' ? info.last_updated : {};
        // Capture any other top-level keys
        const otherInfo = Object.entries(info).filter(([key]) => key !== 'files' && key !== 'last_updated');
        
        // Don't render the section if only files were present (which we are now hiding)
        if (Object.keys(lastUpdated).length === 0 && otherInfo.length === 0) return null;

        return (
            <Paper
                variant="outlined"
                sx={{
                    p: isMobile ? 1.5 : 2,
                    mb: 2, // Add margin bottom
                    bgcolor: isDarkMode ? 'var(--card-bg-color-elevated)' : alpha(theme.palette.grey[50], 0.5),
                    borderColor: theme.palette.divider,
                    borderRadius: '8px' // Match other paper elements
                }}
            >
                <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 2, pb: 1, borderBottom: `1px solid ${theme.palette.divider}` }}>
                    Source Details
                </Typography>

                {/* Files section completely removed */}

                {Object.keys(lastUpdated).length > 0 && (
                    <Box mb={otherInfo.length > 0 ? 2 : 0}> {/* Add margin if other info follows */}
                        <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
                            Last Updated
                        </Typography>
                        <List dense disablePadding>
                            {Object.entries(lastUpdated).map(([file, timestamp]) => (
                                <ListItem 
                                    key={file} 
                                    disableGutters // Remove default padding
                                    sx={{
                                        py: 0.75, // Increase vertical padding slightly
                                        px: 0, // No horizontal padding on item itself
                                        display: 'flex', 
                                        justifyContent: 'space-between', 
                                        alignItems: 'center',
                                        borderBottom: `1px dashed ${alpha(theme.palette.divider, 0.4)}`, // Make dashed border subtler
                                        '&:last-child': { borderBottom: 'none' } 
                                    }}
                                >
                                    {/* Remove colon, make filename slightly bolder */}
                                    <Typography variant="body2" sx={{ fontWeight: 500, color: 'text.primary' }}> 
                                        {file}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary" sx={{ fontFamily: 'monospace', textAlign: 'right' }}>
                                        {typeof timestamp === 'string' ? 
                                            new Date(timestamp).toLocaleString() : 
                                            String(timestamp)}
                                    </Typography>
                                </ListItem>
                            ))}
                        </List>
                    </Box>
                )}

                {otherInfo.length > 0 && (
                     <Box>
                        <Typography variant="subtitle2" sx={{ mb: 0.5, color: 'text.secondary' }}>Other Info:</Typography>
                        <List dense disablePadding>
                           {otherInfo.map(([key, value]) => (
                                <ListItem key={key} sx={{ py: 0.25, pl: 1, display: 'flex', justifyContent: 'space-between' }}>
                                    <Typography variant="body2" sx={{ mr: 1, textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}:</Typography>
                                    <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'right' }}>
                                        {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                    </Typography>
                                </ListItem>
                            ))}
                        </List>
                     </Box>
                )}
            </Paper>
        );
     };
     // --- END NEW Helper ---

     return (
       <Box sx={{ p: theme.spacing(2), pb: isMobile ? theme.spacing(16) : theme.spacing(2) }}>
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
                     primary={<Typography component="div">{credit.credit}</Typography>}
                 />
               </ListItem>
             ))}
           </List>
          </>
         )}

         {/* Source Information (Use new renderer instead of Accordion) */}
         {/* {renderJsonData('Source Information', sourceInfo)} */}
         {hasSourceInfo && renderStructuredSourceInfo(sourceInfo)}

         {/* Completeness Info - Use score - ONLY IN DEV */}
         {import.meta.env.DEV && hasScore && (
            <Box sx={{ mt: 3 }}>
              <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 600 }}>Completeness Score</Typography>
              <Chip 
                label={`${(score * 100).toFixed(0)}%`} 
                color="success" 
                variant="filled"
                size="small"
                sx={{ justifyContent: 'flex-start' }}
              />
              {/* Remove old data_completeness iteration */}
            </Box>
         )}

         {/* Entry Timestamps */}
         {hasEntryInfo && (
           <Box sx={{ mt: 3, pt: 2, borderTop: hasCredits || hasSourceInfo || hasScore ? `1px solid ${theme.palette.divider}` : 'none' }}>
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
      ref={ref} 
      className={`word-details-container ${theme.palette.mode}`}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%', 
        overflow: 'hidden', 
        bgcolor: 'background.default',
        width: '100%', 
        maxWidth: 'none', 
        p: 0, // No padding on the outermost container
        m: 0 
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
      <Box sx={{ 
          display: 'flex', 
          flexDirection: 'row', // Always row now
          flexGrow: 1, 
          overflow: 'hidden',
          height: isMobile ? '0px' : 'auto' // Ensure this line is correctly added
        }}>
        
          <Tabs
            orientation="vertical"
            variant="scrollable"
            value={activeTab}
            onChange={handleTabChange}
            aria-label="Word details sections"
            sx={{
              borderRight: 1,
              borderColor: 'divider',
              // Adjust minWidth for mobile
              minWidth: isMobile ? 100 : 160, // *** Reduced mobile minWidth ***
              bgcolor: 'background.paper',
              flexShrink: 0,
              // Adjust tab padding and font size for mobile if needed
               '& .MuiTab-root': {
                    textTransform: 'none',
                    fontWeight: theme.typography.fontWeightRegular,
                    fontSize: theme.typography.pxToRem(isMobile ? 12 : 14), // Smaller font on mobile
                    minHeight: isMobile ? 40 : 48, // *** Reduced mobile minHeight ***
                    justifyContent: 'flex-start',
                    pl: isMobile ? 1 : 2, // Less padding on mobile
                    pr: isMobile ? 0.5 : 1, // Less padding on mobile
                    py: isMobile ? 0.5 : 1, // Adjust vertical padding if needed
                    minWidth: isMobile ? 90 : 'auto', // *** Reduced mobile minWidth ***
                    '&.Mui-selected': { /* Existing inner sx */ },
                    '&:hover': { /* Existing inner sx */ },
                  },
                  '& .MuiTabs-indicator': { /* Existing inner sx */ },
            }}
          >
            {/* Tabs remain the same */} 
            <Tab label="Definitions" value="definitions" />
            <Tab label="Relations" value="relations" />
            <Tab label="Etymology" value="etymology" />
            <Tab label="Forms" value="forms" />
            <Tab label="Sources" value="sources" />
          </Tabs>

        {/* Tab Content Panel (Scrollable) */}
        <Box
          role="tabpanel"
          hidden={false} // Keep it always rendered for simplicity
          sx={{ /* Existing sx props */
            flexGrow: 1, 
            overflowY: 'auto', 
            p: isMobile ? 1 : 2, // *** Reduced mobile padding ***
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

    </Box> // End Main Wrapper Box
  );
}); // End forwardRef

// Export the component with memoization
export default React.memo(WordDetailsComponent);