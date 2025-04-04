import React, { useCallback, useState, useEffect } from 'react';
import { WordInfo, RelatedWord, Definition, RawEtymology, EtymologyComponent, Relation, EtymologyTree, EtymologyNode } from '../types';
// import { fetchWordNetwork } from '../api/wordApi';
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
import Tooltip from '@mui/material/Tooltip';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';

// Add d3 import here at the top
import * as d3 from 'd3';
import EtymologyTreeGraph from './EtymologyTreeGraph'; // Import the new component

// MUI Icons
// import VolumeUpIcon from '@mui/icons-material/VolumeUp';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import StopCircleIcon from '@mui/icons-material/StopCircle'; // Icon for stop button

interface WordDetailsProps {
  wordInfo: WordInfo;
  etymologyTree: EtymologyTree | null;
  isLoadingEtymology: boolean;
  etymologyError: string | null;
  onWordLinkClick: (word: string) => void;
  onEtymologyNodeClick: (node: any) => void;
  showMetadata: boolean;
  setShowMetadata: React.Dispatch<React.SetStateAction<boolean>>;
}

// Helper function to format relation type names
function formatRelationType(type: string): string {
  return type
    .replace(/_/g, ' ') // Replace underscores with spaces
    .replace(/\b\w/g, char => char.toUpperCase()); // Capitalize first letter of each word
}

// Define key graph colors locally for styling
const graphColors = {
  main: "#1d3557",
  root: "#e63946",
  derived: "#2a9d8f",
  synonym: "#457b9d",
  antonym: "#f77f00",
  variant: "#f4a261",
  related: "#fcbf49",
  // Add other colors from your WordGraph getNodeColor if needed
  associated: "#adb5bd",
  default: "#6c757d"
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
    borderBottom: `1px solid ${theme.palette.divider}`, // Use divider for separation
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
      ? 'rgba(255, 255, 255, .03)'
      : 'rgba(0, 0, 0, .02)',
  '&:hover': {
     backgroundColor:
        theme.palette.mode === 'dark'
          ? 'rgba(255, 255, 255, .05)'
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
  backgroundColor: 'transparent',
}));

const ExpandMoreIcon = () => <Typography sx={{ transform: 'rotate(90deg)', lineHeight: 0 }}>▶</Typography>;
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0 }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0 }}>⏹️</Typography>;

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  etymologyTree,
  isLoadingEtymology,
  etymologyError,
  onWordLinkClick,
  onEtymologyNodeClick,
  showMetadata,
  setShowMetadata
}) => {
  const theme = useTheme(); // Get theme object
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md')); // Use 'md' breakpoint for vertical tabs
  const isDarkMode = theme.palette.mode === 'dark';

  const [activeTab, setActiveTab] = useState<string>('definitions'); // Use string for tab value
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

  // State for Accordion expansion
  const [expandedDefinition, setExpandedDefinition] = useState<string | false>(false);
  const [expandedEtymology, setExpandedEtymology] = useState<string | false>(false);

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
  }, [wordInfo]);

  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => {
    setActiveTab(newValue);
  };

  // Handlers for Accordion expansion
  const handleDefinitionChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedDefinition(isExpanded ? panel : false);
  };
  const handleEtymologyChange = (panel: string) => (event: React.SyntheticEvent, isExpanded: boolean) => {
    setExpandedEtymology(isExpanded ? panel : false);
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

    const headerBgColor = isDarkMode
      ? alpha(graphColors.main, 0.6)
      : alpha(graphColors.main, 0.1);
    const effectiveHeaderBg = theme.palette.augmentColor({ color: { main: headerBgColor } });
    const headerTextColor = effectiveHeaderBg.contrastText;

    return (
      // Use theme spacing for padding
      <Box sx={{ bgcolor: headerBgColor, color: headerTextColor, pt: theme.spacing(3), pb: theme.spacing(3), pl: theme.spacing(3), pr: theme.spacing(2) }}>
        {/* Lemma and Audio Button */}
        <Stack direction="row" spacing={1} alignItems="flex-start" sx={{ mb: theme.spacing(1.5) }}>
          <Typography variant="h3" component="h1" sx={{ fontWeight: 700, flexGrow: 1, lineHeight: 1.2 }}>
            {wordInfo.lemma}
          </Typography>
          {hasAudio && (
            <IconButton
              onClick={playAudio}
              size="medium"
              title={isAudioPlaying ? "Stop Audio" : "Play Audio"}
              sx={{ color: headerTextColor, mt: 0.5, '&:hover': { bgcolor: alpha(headerTextColor, 0.1) } }}
            >
              {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
            </IconButton>
          )}
        </Stack>

        {/* Pronunciation (IPA) */}
        {ipaPronunciation && (
          <Typography variant="h6" sx={{ color: alpha(headerTextColor, 0.85), fontStyle: 'italic', mb: theme.spacing(1.5), pl: theme.spacing(0.5) }}>
            /{ipaPronunciation.value}/
          </Typography>
        )}

        {/* Baybayin */}
        {wordInfo.has_baybayin && wordInfo.baybayin_form && (
          <Box sx={{ my: theme.spacing(2) }}>
            <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.75), display: 'block', mb: 0.5 }}>
              Baybayin Script
            </Typography>
            <Typography
              variant="h4"
              sx={{
                fontFamily: 'Noto Sans Baybayin, sans-serif',
                p: theme.spacing(1),
                bgcolor: alpha(headerTextColor, 0.08),
                borderRadius: 1,
                display: 'inline-block',
                lineHeight: 1,
              }}
            >
              {wordInfo.baybayin_form}
            </Typography>
          </Box>
        )}

        {/* Tags */}
        {tags.length > 0 && (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: theme.spacing(2) }}>
            {tags.map((tag) => (
              <Chip
                key={tag}
                label={tag}
                size="small"
                sx={{
                  color: alpha(headerTextColor, 0.9),
                  borderColor: alpha(headerTextColor, 0.5),
                  bgcolor: 'transparent',
                  '& .MuiChip-label': { fontWeight: 500 },
                  height: 'auto', // Allow chip height to adjust
                  padding: theme.spacing(0.25, 0.75)
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
      return <Typography sx={{ p: 2, fontStyle: 'italic' }}>No definitions available.</Typography>;
    }

    return (
      <Box>
        {Object.entries(wordInfo.standardized_pos || {}).map(([pos, definitions]) => (
          <Accordion key={pos} sx={{ mb: 1, backgroundColor: theme === 'dark' ? '#2c2c2c' : '#f9f9f9' }} 
                     expanded={expandedDefinition === pos} onChange={handleDefinitionChange(pos)}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls={`${pos}-content`} id={`${pos}-header`}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>{pos}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <List dense disablePadding>
                {(definitions as Definition[]).map((def, index) => (
                  <ListItem key={index} sx={{ display: 'block', borderBottom: index < definitions.length - 1 ? '1px dashed #ccc' : 'none', pb: 1, mb: 1 }}>
                    <Typography variant="body1" gutterBottom>
                      {index + 1}. {def.definition}
                    </Typography>
                    {def.examples && def.examples.length > 0 && (
                      <Box sx={{ pl: 2, my: 0.5 }}>
                        {def.examples.map((ex, exIdx) => {
                          const exampleData = ex as { example: string; translation?: string };
                          return (
                            <Typography key={exIdx} variant="caption" display="block" sx={{ fontStyle: 'italic' }}>
                              "{exampleData.example}"{exampleData.translation ? ` - ${exampleData.translation}` : ''}
                            </Typography>
                          );
                        })}
                      </Box>
                    )}
                    {def.synonyms && def.synonyms.length > 0 && (
                      <Typography variant="caption" display="block">
                        <strong>Synonyms:</strong> {def.synonyms.join(', ')}
                      </Typography>
                    )}
                    {def.antonyms && def.antonyms.length > 0 && (
                      <Typography variant="caption" display="block">
                        <strong>Antonyms:</strong> {def.antonyms.join(', ')}
                      </Typography>
                    )}
                    {showMetadata && def.metadata && (
                       <Box sx={{ mt: 1, pt: 1, borderTop: '1px dashed grey', fontSize: '0.8rem', color: 'text.secondary' }}>
                           {Object.entries(def.metadata).map(([key, value]) => (
                               <Typography key={key} variant="caption" display="block">{key}: {String(value)}</Typography>
                           ))}
                       </Box>
                    )}
                  </ListItem>
                ))}
              </List>
            </AccordionDetails>
          </Accordion>
        ))}
        {Object.keys(wordInfo.standardized_pos || {}).length === 0 && (
          <Typography sx={{ fontStyle: 'italic', color: 'text.secondary' }}>No definitions available.</Typography>
        )}
      </Box>
    );
  };

  const renderRelationsTab = () => {
    const hasRelations = 
        (wordInfo.outgoing_relations && wordInfo.outgoing_relations.length > 0) ||
        (wordInfo.incoming_relations && wordInfo.incoming_relations.length > 0);

    if (!hasRelations) {
        return <Typography sx={{ p: 2, fontStyle: 'italic' }}>No relations available.</Typography>;
    }

    // Group relations by type
    const outgoingGrouped: { [key: string]: Relation[] } = {};
    (wordInfo.outgoing_relations || []).forEach(rel => {
        if (!outgoingGrouped[rel.relation_type]) outgoingGrouped[rel.relation_type] = [];
        outgoingGrouped[rel.relation_type].push(rel);
    });

    const incomingGrouped: { [key: string]: Relation[] } = {};
    (wordInfo.incoming_relations || []).forEach(rel => {
        if (!incomingGrouped[rel.relation_type]) incomingGrouped[rel.relation_type] = [];
        incomingGrouped[rel.relation_type].push(rel);
    });

    const renderRelationGroup = (title: string, relations: { [key: string]: Relation[] }) => (
      <Box sx={{ mb: 2 }}>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        {Object.entries(relations).length === 0 ? (
          <Typography variant="body2" sx={{ fontStyle: 'italic' }}>None</Typography>
        ) : (
          Object.entries(relations).map(([type, rels]) => (
            <Box key={type} sx={{ mb: 1.5 }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>{formatRelationType(type)}</Typography>
              <List dense disablePadding>
                {rels.map(rel => (
                  <ListItem key={rel.id} disableGutters>
                    <Link 
                      component="button" 
                      onClick={() => onWordLinkClick(rel.target_word?.lemma || rel.source_word?.lemma || '' )} 
                      disabled={!(rel.target_word?.lemma || rel.source_word?.lemma)}
                      sx={{ textAlign: 'left' }}
                    >
                       {rel.target_word?.lemma || rel.source_word?.lemma}
                    </Link>
                    {/* Display relation metadata if needed */}
                  </ListItem>
                ))}
              </List>
            </Box>
          ))
        )}
      </Box>
    );

    return (
        <Box sx={{ p: 2 }}>
            {renderRelationGroup('Outgoing Relations', outgoingGrouped)}
            <Divider sx={{ my: 2 }} />
            {renderRelationGroup('Incoming Relations', incomingGrouped)}
        </Box>
    );
  };

  // Refactor renderEtymologyTab to use RawEtymology and components
  const renderEtymologyTab = () => {
    // API call returns RawEtymology[], check wordInfo for this field
    const etymologies = wordInfo.etymologies as RawEtymology[] | undefined; 

    if (!etymologies || etymologies.length === 0) {
      return <Typography sx={{ p: 2, fontStyle: 'italic' }}>No etymology available.</Typography>;
    }

    return (
      <Box>
        {etymologies.map((etym, index) => (
          <Accordion key={index} sx={{ mb: 1, backgroundColor: theme === 'dark' ? '#2c2c2c' : '#f9f9f9' }} 
                     expanded={expandedEtymology === `etym-${index}`} onChange={handleEtymologyChange(`etym-${index}`)}>
            <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls={`etym-${index}-content`} id={`etym-${index}-header`}>
              <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>Etymology {wordInfo.etymologies.length > 1 ? index + 1 : ''}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              {etym.components && etym.components.length > 0 && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="body2" component="div" dangerouslySetInnerHTML={{ __html: etym.components.map(c => c.html).join('') }} />
                </Box>
              )}
              {/* Fallback or additional text if components are not exhaustive */}
              {!etym.components && etym.raw_text && (
                <Typography variant="body2" sx={{ fontStyle: 'italic' }}>{etym.raw_text}</Typography>
              )}
              {showMetadata && etym.metadata && (
                 <Box sx={{ mt: 1, pt: 1, borderTop: '1px dashed grey', fontSize: '0.8rem', color: 'text.secondary' }}>
                     {Object.entries(etym.metadata).map(([key, value]) => (
                         <Typography key={key} variant="caption" display="block">{key}: {String(value)}</Typography>
                     ))}
                 </Box>
              )}
            </AccordionDetails>
          </Accordion>
        ))}

        {/* Render Etymology Tree Graph */} 
        <Typography variant="h6" sx={{ mt: 3, mb: 1 }}>Etymology Tree</Typography>
        {isLoadingEtymology && <CircularProgress size={24} />}
        {etymologyError && <Typography color="error">Error loading tree: {etymologyError}</Typography>}
        {!isLoadingEtymology && !etymologyError && etymologyTree && (
          <EtymologyTreeGraph 
            etymologyTree={etymologyTree} 
            onEtymologyNodeClick={onEtymologyNodeClick} 
          />
        )}
        {!isLoadingEtymology && !etymologyError && !etymologyTree && wordInfo.etymologies && wordInfo.etymologies.length > 0 && (
          <Typography sx={{ fontStyle: 'italic', color: 'text.secondary', mt: 1 }}>Etymology tree visualization not available.</Typography>
        )}
      </Box>
    );
  };

  const renderCreditsTab = () => {
    if (!wordInfo.credits || wordInfo.credits.length === 0) {
      return <Typography sx={{ p: 2, fontStyle: 'italic' }}>No credits available.</Typography>;
    }
    return (
      <List dense sx={{ p: 1 }}>
        {wordInfo.credits.map((credit, index) => (
          <React.Fragment key={credit.id || index}>
            <ListItem>
              <ListItemText primary={credit.credit} />
            </ListItem>
            {index < wordInfo.credits!.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>
    );
  };

  // --- Main Return --- 

  if (!wordInfo) {
    return <Typography sx={{ p: 3 }}>Loading word details...</Typography>; 
  }

  const tabs = [
    { label: 'Definitions', value: 'definitions', content: renderDefinitionsTab },
    { label: 'Etymology', value: 'etymology', content: renderEtymologyTab },
    { label: 'Relations', value: 'relations', content: renderRelationsTab },
    { label: 'Credits', value: 'credits', content: renderCreditsTab },
  ];

  const activeTabContent = tabs.find(tab => tab.value === activeTab)?.content();

  return (
    <Paper elevation={2} sx={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {renderHeader()}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange} 
          aria-label="Word details tabs"
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile // Ensure scroll buttons appear on mobile if needed
        >
          {tabs.map(tab => (
            <Tab key={tab.value} label={tab.label} value={tab.value} />
          ))}
          {/* Metadata Toggle Button */} 
          <Tooltip title={`${showMetadata ? 'Hide' : 'Show'} Metadata`}>
            <IconButton 
               onClick={() => setShowMetadata(!showMetadata)} 
               size="small" 
               sx={{ ml: 'auto', mr: 1}} 
            >
              <InfoOutlinedIcon />
            </IconButton>
          </Tooltip>
        </Tabs>
      </Box>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 0 }}> {/* Remove padding here, add in render functions */}
        {activeTabContent}
      </Box>
    </Paper>
  );
});

export default WordDetails;