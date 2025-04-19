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
  
  // Helper to safely render potentially dangerous HTML (like definitions)
  const renderSafeHtml = (htmlString: string | undefined | null): React.ReactNode => {
    if (!htmlString) return null;
    // Basic sanitization (consider DOMPurify for robust sanitization if needed)
    const sanitizedHtml = htmlString.replace(/<script.*?>.*?<\/script>/gi, '');
    return <span dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />;
  };

  // Helper to render a clickable word link
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
  
  // --- DEBUG: Comment out other render functions for now --- 
  /*
  const renderRelationsTab = () => {
    const organizedRelations = useMemo(() => {
        if (!wordInfo) return {};

        const allRelations: Record<string, RelatedWord[]> = {};

        const addRelation = (type: string, word: RelatedWord | undefined) => {
            if (!word) return;
            if (!allRelations[type]) {
                allRelations[type] = [];
            }
            // Avoid duplicates based on ID
            if (!allRelations[type].some(existing => existing.id === word.id)) {
                allRelations[type].push(word);
            }
        };

        // Process incoming relations
        wordInfo.incoming_relations?.forEach(rel => {
            addRelation(rel.relation_type, rel.source_word);
        });

        // Process outgoing relations
        wordInfo.outgoing_relations?.forEach(rel => {
            addRelation(rel.relation_type, rel.target_word);
        });
        
        // Process semantic network if relations are empty
        if (Object.keys(allRelations).length === 0 && wordInfo.semantic_network) {
            console.log("Processing semantic network for relations...");
            const mainNodeId = wordInfo.id;
            const nodesById = new Map(wordInfo.semantic_network.nodes.map(n => [n.id, n]));

            wordInfo.semantic_network.links.forEach(link => {
                const sourceNode = nodesById.get(link.source as number);
                const targetNode = nodesById.get(link.target as number);
                
                if (sourceNode && targetNode) {
                    const relationType = link.type || 'related'; // Default if type missing
                    let relatedWord: RelatedWord | undefined;
                    
                    // Check if the main node is involved
                    if (targetNode.id === mainNodeId) { // Incoming relation
                        relatedWord = {
                            id: sourceNode.id,
                            lemma: sourceNode.label || sourceNode.word || 'Unknown',
                            language_code: sourceNode.language,
                            has_baybayin: sourceNode.has_baybayin,
                            baybayin_form: sourceNode.baybayin_form
                        };
                        addRelation(relationType, relatedWord);
                        
                    } else if (sourceNode.id === mainNodeId) { // Outgoing relation
                         relatedWord = {
                            id: targetNode.id,
                            lemma: targetNode.label || targetNode.word || 'Unknown',
                            language_code: targetNode.language,
                            has_baybayin: targetNode.has_baybayin,
                            baybayin_form: targetNode.baybayin_form
                        };
                        addRelation(relationType, relatedWord);
                    }
                }
            });
        }

        // Combine root_affixations and affixed_affixations
        const affixations = [
          ...(wordInfo.root_affixations || []).map(a => ({...a, direction: 'derived_from_root' as const})),
          ...(wordInfo.affixed_affixations || []).map(a => ({...a, direction: 'affixed_to_root' as const}))
        ];

        affixations.forEach(affix => {
            if (affix.direction === 'derived_from_root') {
                addRelation(affix.affix_type, affix.affixed_word);
            } else if (affix.direction === 'affixed_to_root') {
                 addRelation(`affixed_with_${affix.affix_type}`, affix.root_word);
            }
        });

        // Add root word if present
        if (wordInfo.root_word) {
          addRelation('root_word', wordInfo.root_word);
        }
        // Add derived words if present
        if (wordInfo.derived_words && wordInfo.derived_words.length > 0) {
          wordInfo.derived_words.forEach(dw => addRelation('derived_word', dw));
        }

        // Sort relation types (optional)
        const sortedTypes = Object.keys(allRelations).sort();
        const sortedRelations: Record<string, RelatedWord[]> = {};
        sortedTypes.forEach(type => {
          sortedRelations[type] = allRelations[type];
        });

        return sortedRelations;
    }, [wordInfo]);

    if (!wordInfo || Object.keys(organizedRelations).length === 0) {
      return <Typography>No relations available.</Typography>;
    }

    return (
      <Box>
        {Object.entries(organizedRelations).map(([type, words]) => (
          <StyledAccordion key={type} defaultExpanded={['synonym', 'antonym', 'root_word', 'derived_word'].includes(type)}> 
            <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography sx={{ width: '40%', flexShrink: 0, fontWeight: 'medium', color: 'var(--section-header-color)' }}>
                {formatRelationType(type)}
              </Typography>
              <Chip label={words.length} size="small" sx={{ bgcolor: relationColors[type] || graphColors.default, color: isColorLight(relationColors[type] || graphColors.default) ? '#000' : '#fff' }}/>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                {words.map((word, index) => (
                  word ? renderWordLink(word, `${type}-${index}`) : null
                ))}
              </Stack>
            </StyledAccordionDetails>
          </StyledAccordion>
        ))}
      </Box>
    );
  };

  const renderFormsAndTemplatesTab = () => {
    const hasForms = wordInfo?.forms && wordInfo.forms.length > 0;
    const hasTemplates = wordInfo?.templates && wordInfo.templates.length > 0;
    const hasIdioms = wordInfo?.idioms && (Array.isArray(wordInfo.idioms) ? wordInfo.idioms.length > 0 : Object.keys(wordInfo.idioms).length > 0);

    if (!hasForms && !hasTemplates && !hasIdioms) {
      return <Typography>No forms, templates, or idioms available.</Typography>;
    }

    // Prepare idioms data (handle both array and object formats)
    let idiomsList: Idiom[] = [];
    if (wordInfo.idioms) {
      if (Array.isArray(wordInfo.idioms)) {
        idiomsList = wordInfo.idioms;
      } else if (typeof wordInfo.idioms === 'object') {
        // Convert object to array if needed, assuming keys are identifiers/phrases
        idiomsList = Object.entries(wordInfo.idioms).map(([key, value]) => ({
          phrase: key,
          ...(typeof value === 'object' ? value : { meaning: String(value) }) // Handle simple string meanings
        }));
      }
    }

    return (
      <Box>
        {hasForms && (
          <Box mb={3}>
            <Typography variant="h6" component="h3" gutterBottom sx={{ color: 'var(--section-header-color)' }}>Forms</Typography>
            <List dense disablePadding>
              {wordInfo.forms!.map((form, index) => (
                <ListItem key={form.id || index} disableGutters>
                  <ListItemText 
                    primary={form.form} 
                    secondary={form.tags ? Object.entries(form.tags).map(([k,v]) => `${k}:${v}`).join(', ') : null}
                    sx={{ '& .MuiListItemText-primary': { color: 'var(--text-color)' }, '& .MuiListItemText-secondary': { color: 'var(--secondary-text-color)' } }}
                   />
                  {form.is_canonical && <Chip label="Canonical" size="small" color="primary" variant="outlined" sx={{ ml: 1 }}/>}
                  {form.is_primary && <Chip label="Primary" size="small" color="secondary" variant="outlined" sx={{ ml: 1 }}/>}
                </ListItem>
              ))}
            </List>
          </Box>
        )}

        {hasTemplates && (
          <Box mb={3}>
            <Typography variant="h6" component="h3" gutterBottom sx={{ color: 'var(--section-header-color)' }}>Templates</Typography>
            <List dense disablePadding>
              {wordInfo.templates!.map((template, index) => (
                <ListItem key={template.id || index} disableGutters>
                  <ListItemText 
                    primary={template.template_name} 
                    secondary={template.expansion || (template.args ? `Args: ${JSON.stringify(template.args)}` : null)}
                    sx={{ '& .MuiListItemText-primary': { color: 'var(--text-color)' }, '& .MuiListItemText-secondary': { color: 'var(--secondary-text-color)', whiteSpace: 'pre-wrap' } }}
                   />
                </ListItem>
              ))}
            </List>
          </Box>
        )}
        
        {hasIdioms && (
          <Box>
            <Typography variant="h6" component="h3" gutterBottom sx={{ color: 'var(--section-header-color)' }}>Idioms & Phrases</Typography>
            <List dense disablePadding>
              {idiomsList.map((idiom, index) => (
                <ListItem key={index} alignItems="flex-start" disableGutters>
                  <ListItemText
                    primary={idiom.phrase || idiom.text}
                    secondary={
                      <Stack spacing={0.5}>
                        {idiom.meaning && <Typography variant="body2" sx={{color: 'var(--secondary-text-color)'}}>{idiom.meaning}</Typography>}
                        {idiom.example && <Typography variant="caption" sx={{color: 'var(--secondary-text-color)', fontStyle: 'italic'}}>e.g., {idiom.example}</Typography>}
                        {idiom.source && <Typography variant="caption" sx={{color: 'var(--secondary-text-color)'}}>Source: {idiom.source}</Typography>}
                      </Stack>
                    }
                     sx={{ '& .MuiListItemText-primary': { color: 'var(--text-color)', fontWeight: 'medium' } }}
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
    const hasEtymologies = wordInfo?.etymologies && wordInfo.etymologies.length > 0;
    const hasTreeData = etymologyTree && etymologyTree.nodes && etymologyTree.nodes.length > 0;

    const renderEtymologyTreeVisualization = (hasTreeData: boolean) => {
      const containerRef = React.useRef<HTMLDivElement>(null);
      const svgRef = React.useRef<SVGSVGElement>(null);

      useEffect(() => {
        if (!hasTreeData || !containerRef.current || !svgRef.current || !etymologyTree) return;

        const container = containerRef.current;
        const svgElement = svgRef.current;
        const width = container.clientWidth;
        const height = 500; // Fixed height or calculate based on data

        d3.select(svgElement).selectAll("*").remove(); // Clear previous render

        const svg = d3.select(svgElement)
          .attr("width", width)
          .attr("height", height)
          .attr("viewBox", [-width / 2, -height / 2, width, height])
          .style("max-width", "100%")
          .style("height", "auto")
          .style("background-color", isDarkMode ? "#333" : "#f8f8f8")
          .style("border-radius", "4px");
          
        const g = svg.append("g");

        // --- D3 Tree Layout ---
        const root = d3.stratify<any>()
            .id((d: any) => d.id)
            // Use parentId if available, otherwise assume root if parentId is null/undefined
            .parentId((d: any) => d.parentId)
            (etymologyTree.nodes);
            
        if (!root) {
          console.error("Failed to create root for tree layout");
          return;
        }

        const treeLayout = d3.tree().size([height, width - 160]); // Adjust width for labels
        treeLayout(root);
        
        // Links
        g.append("g")
          .attr("fill", "none")
          .attr("stroke", isDarkMode ? "#777" : "#ccc")
          .attr("stroke-opacity", 0.6)
          .attr("stroke-width", 1.5)
          .selectAll("path")
          .data(root.links())
          .join("path")
          .attr("d", d3.linkHorizontal()
              .x((d: any) => d.y) // Use y for horizontal position
              .y((d: any) => d.x) as any); // Use x for vertical position
              
        // Nodes
        const node = g.append("g")
          .selectAll("g")
          .data(root.descendants())
          .join("g")
    }
    
    // If only tree data is available, attempt to render it
    if (hasEtymologyTreeData && !hasWordEtymologies) {
       return (
         <Box sx={{ p: theme.spacing(2) }}>
           {renderEtymologyTreeVisualization(hasEtymologyTreeData)} {/* Pass flag */} 
         </Box>
       );
    }

    // Fallback if somehow neither condition above was met (shouldn't happen due to initial checks)
    return null;
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
             <Paper 
    variant="outlined" 
    sx={{ 
      p: 1.5, 
      bgcolor: isDarkMode ? 'rgba(30, 40, 60, 0.3)' : alpha(theme.palette.grey[500], 0.05), 
      borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : undefined,
      overflowX: 'auto' 
    }}
>
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
                    sx={{ 
                      justifyContent: 'flex-start',
                      ...(isDarkMode && {
                        bgcolor: value 
                          ? alpha(theme.palette.success.main, 0.3) // More visible success background
                          : 'transparent',
                        borderColor: value 
                          ? alpha(theme.palette.success.main, 0.7) 
                          : alpha(theme.palette.grey[500], 0.3),
                        color: value 
                          ? theme.palette.success.light 
                          : alpha(theme.palette.grey[300], 0.9)
                      })
                    }}
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
            
            // Update the tab content container in wide screen layout
            const tabContentWide = (
              <Box sx={{ 
                flexGrow: 1, 
                overflowY: 'auto', 
                position: 'relative', 
                bgcolor: isDarkMode ? 'var(--card-bg-color-elevated)' : 'background.default',
                width: 'calc(100% - 160px)',
                maxWidth: '100%',
                border: 'none',
                '& .MuiPaper-root': {
                  borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : theme.palette.divider,
                },
                '& .MuiBox-root': {
                  borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.08)' : theme.palette.divider,
                }
              }}>
              <Box sx={{ 
                px: theme.spacing(3), 
                py: theme.spacing(2), 
                width: '100%', 
                maxWidth: '100%',
                border: 'none' 
              }}>
                {activeTab === 'definitions' && renderDefinitionsTab()}
                {activeTab === 'relations' && renderRelationsTab()}
                {activeTab === 'forms_templates' && renderFormsAndTemplatesTab()}
                {activeTab === 'etymology' && renderEtymologyTab()}
                {activeTab === 'sources-info' && renderSourcesInfoTab()}
                {activeTab === 'metadata' && renderSourcesInfoTab()}
              </Box>
            </Box>
            );
            
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
                     borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'divider',
                     minWidth: isWideScreen ? 160 : 'auto',
                     alignItems: isWideScreen ? 'flex-start' : 'center',
                     margin: 0,
                     padding: 0,
                     height: '100%',
                     bgcolor: isDarkMode ? 'var(--card-bg-color)' : 'transparent',
                     '& .MuiTab-root': { 
                       minHeight: 48, 
                       alignItems: 'flex-start',
                       textAlign: 'left',
                       px: isWideScreen ? 2 : 1,
                       color: isDarkMode ? 'var(--text-color)' : 'inherit',
                     },
                     '& .Mui-selected': {
                       color: isDarkMode ? 'var(--primary-color) !important' : 'primary.main !important',
                     },
                     '& .MuiTabs-indicator': {
                       backgroundColor: isDarkMode ? 'var(--primary-color)' : 'primary.main',
                     }
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
        bgcolor: isDarkMode ? 'var(--card-bg-color)' : 'background.paper',
        color: isDarkMode ? 'var(--text-color)' : 'text.primary',
        overflow: 'hidden',
        maxWidth: '100%',
        border: 'none',
        '& .MuiPaper-root': {
          borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : undefined
        }
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
              width: '100%',
              border: 'none'
            }}>
               <Box sx={{ 
                 display: 'flex', 
                 flexDirection: 'column',
                 minWidth: 160,
                 maxWidth: 160,
                 borderRight: 1,
                 borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'divider',
                 bgcolor: isDarkMode ? 'var(--card-bg-color)' : 'background.paper'
               }}>
                 {tabs}
               </Box>
              {tabContentWide}
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
               bgcolor: isDarkMode ? 'var(--card-bg-color-elevated)' : 'background.default',
              width: '100%',
              border: 'none',
              '& .MuiPaper-root': {
                borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.05)' : theme.palette.divider,
              },
              '& .MuiBox-root': {
                borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.08)' : theme.palette.divider,
              }
            }}>
              <Box sx={{ 
                px: theme.spacing(3), 
                py: theme.spacing(2), 
                width: '100%', 
                maxWidth: '100%',
                border: 'none'
              }}>
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