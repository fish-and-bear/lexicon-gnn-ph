import React, { useCallback, useState, useEffect } from 'react';
import { WordInfo, RelatedWord, Definition, RawEtymology, EtymologyComponent, Relation } from '../types';
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
  showMetadata: boolean;
  setShowMetadata: (show: boolean) => void;
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
  const [expandedDefinition, setExpandedDefinition] = useState<number | false>(false);
  const [expandedEtymology, setExpandedEtymology] = useState<number | false>(false);

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
  const handleDefinitionAccordionChange = (panelId: number) => 
    (event: React.SyntheticEvent, isExpanded: boolean) => {
      setExpandedDefinition(isExpanded ? panelId : false);
  };
  const handleEtymologyAccordionChange = (panelId: number) => 
    (event: React.SyntheticEvent, isExpanded: boolean) => {
      setExpandedEtymology(isExpanded ? panelId : false);
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
        {wordInfo.definitions.map((def, index) => (
          <StyledAccordion 
            key={def.id || index} 
            expanded={expandedDefinition === (def.id || index)}
            onChange={handleDefinitionAccordionChange(def.id || index)}
          >
            <StyledAccordionSummary 
              expandIcon={<ExpandMoreIcon />} 
              aria-controls={`definition-${def.id}-content`} 
              id={`definition-${def.id}-header`}
            >
              <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%' }}>
                <Typography sx={{ flexShrink: 0, fontWeight: 500 }}>{index + 1}.</Typography>
                {/* Use standardized_pos object */} 
                {def.standardized_pos && (
                  <Chip 
                    label={def.standardized_pos.name_en || def.standardized_pos.code}
                    size="small" 
                    variant="outlined" 
                    sx={{ 
                      height: 'auto', 
                      '& .MuiChip-label': { 
                        padding: '1px 6px', 
                        fontSize: '0.75rem', 
                        lineHeight: 1.4 
                      }, 
                      mr: 1 
                    }}
                  />
                )}
                {/* Display definition text, ensure it's not cut off */} 
                <Typography sx={{ flexGrow: 1, textAlign: 'left' }}>
                  {def.definition_text || def.text} {/* Use definition_text or text */} 
                </Typography>
              </Stack>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              {/* Render examples, usage notes, tags, sources */} 
              {/* Assuming they are now arrays from WordInfo type */} 
              {def.examples && def.examples.length > 0 && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="subtitle2" gutterBottom>Examples:</Typography>
                  <List dense disablePadding>
                    {def.examples.map((ex, i) => <ListItemText key={i} primary={`• ${ex}`} sx={{ pl: 2 }}/>)}
                  </List>
                </Box>
              )}
              {def.usage_notes && def.usage_notes.length > 0 && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="subtitle2" gutterBottom>Usage Notes:</Typography>
                  <List dense disablePadding>
                    {def.usage_notes.map((note, i) => <ListItemText key={i} primary={`• ${note}`} sx={{ pl: 2 }}/>)}
                  </List>
                </Box>
              )}
              {def.tags && def.tags.length > 0 && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Tags:</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                    {def.tags.map((tag, i) => <Chip key={i} label={tag} size="small" />)}
                  </Stack>
                </Box>
              )}
              {def.sources && def.sources.length > 0 && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Sources:</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                    {def.sources.map((src, i) => <Chip key={i} label={src} size="small" variant="outlined" />)}
                  </Stack>
                </Box>
              )}
              {/* Add Metadata Toggle */} 
              {showMetadata && (
                <Box sx={{ mt: 2, borderTop: `1px dashed ${theme.palette.divider}`, pt: 1 }}>
                  <Typography variant="caption" display="block">ID: {def.id}</Typography>
                  <Typography variant="caption" display="block">Original POS: {def.original_pos || 'N/A'}</Typography>
                  <Typography variant="caption" display="block">Confidence: {def.confidence_score ?? 'N/A'}</Typography>
                  <Typography variant="caption" display="block">Verified: {def.is_verified ? 'Yes' : 'No'}</Typography>
                  {/* Add other metadata fields */} 
                </Box>
              )}
            </StyledAccordionDetails>
          </StyledAccordion>
        ))}
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
          <StyledAccordion 
            key={etym.id || index} 
            expanded={expandedEtymology === (etym.id || index)}
            onChange={handleEtymologyAccordionChange(etym.id || index)}
          >
            <StyledAccordionSummary 
              expandIcon={<ExpandMoreIcon />} 
              aria-controls={`etymology-${etym.id}-content`} 
              id={`etymology-${etym.id}-header`}
            >
              {/* Maybe show a snippet or first language code? */}
              <Typography sx={{ flexGrow: 1, textAlign: 'left' }}>
                 Etymology {index + 1} {etym.language_codes ? `(${etym.language_codes.split(',')[0]}...)` : ''}
              </Typography>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              {/* Display full etymology_text if available */} 
              {etym.etymology_text && (
                <Typography variant="body2" paragraph sx={{ fontStyle: 'italic' }}>
                   {etym.etymology_text}
                </Typography>
              )}

              {/* Display extracted components */}
              {etym.components && etym.components.length > 0 && (
                <Box sx={{ my: 1.5 }}>
                  <Typography variant="subtitle2" gutterBottom>Components:</Typography>
                  <List dense disablePadding>
                    {etym.components.map((comp, i) => (
                       <ListItemText 
                         key={i} 
                         primary={
                            <Typography component="span" variant="body2">
                              • <Typography component="strong" variant="body2" sx={{ fontWeight: 'bold' }}>{comp.word_part || comp.original_text}</Typography> 
                              {comp.language && ` (${comp.language})`} 
                              {comp.meaning && `: ${comp.meaning}`}
                            </Typography>
                         }
                         sx={{ pl: 2, mb: 0.5 }}/>
                    ))}
                  </List>
                </Box>
              )}
              
              {/* Display sources if available (needs splitting if string) */} 
              {etym.sources && (
                <Box sx={{ mb: 1.5 }}>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>Sources:</Typography>
                  <Stack direction="row" spacing={0.5} useFlexGap flexWrap="wrap">
                    {(typeof etym.sources === 'string' ? etym.sources.split(',').map(s=>s.trim()).filter(Boolean) : etym.sources).map((src, i) => 
                      <Chip key={i} label={src} size="small" variant="outlined" />
                    )}
                  </Stack>
                </Box>
              )}

              {/* Add Metadata Toggle */} 
              {showMetadata && (
                <Box sx={{ mt: 2, borderTop: `1px dashed ${theme.palette.divider}`, pt: 1 }}>
                  <Typography variant="caption" display="block">ID: {etym.id}</Typography>
                  <Typography variant="caption" display="block">Confidence: {etym.confidence_level || 'N/A'}</Typography>
                  <Typography variant="caption" display="block">Verification: {etym.verification_status || 'N/A'}</Typography>
                   {/* Add other metadata fields */}
                </Box>
              )}
            </StyledAccordionDetails>
          </StyledAccordion>
        ))}

        {/* Render Etymology Tree below accordions */} 
        <Box sx={{ mt: 3, p: 1 }}>
          <Typography variant="h6" gutterBottom>Etymology Tree</Typography>
          {isLoadingEtymology && <CircularProgress size={24} sx={{ display: 'block', mx: 'auto' }} />} 
          {etymologyError && <Alert severity="error">{etymologyError}</Alert>}
          {!isLoadingEtymology && !etymologyError && etymologyTree && (
            // Placeholder for tree rendering - Pass onEtymologyNodeClick
            <Box sx={{ minHeight: 200, border: '1px solid', borderColor: 'divider', p: 1, position: 'relative' }}>
              <Typography variant="caption" sx={{ fontStyle: 'italic' }}>Tree rendering placeholder. Use D3 or another library here.</Typography>
              {/* Example: Render simple list of nodes for testing */} 
              {etymologyTree.nodes?.map((node: any) => (
                 <Link key={node.id} component="button" onClick={() => onEtymologyNodeClick(node)} sx={{ display: 'block', my: 0.5 }}>
                   {node.label || node.name} ({node.language})
                 </Link>
              ))}
            </Box>
          )}
           {!isLoadingEtymology && !etymologyError && !etymologyTree && (
             <Typography sx={{ fontStyle: 'italic' }}>No tree data.</Typography>
           )}
        </Box>
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
          <IconButton 
             onClick={() => setShowMetadata(!showMetadata)} 
             size="small" 
             sx={{ ml: 'auto', mr: 1}} 
             title={showMetadata ? "Hide Metadata" : "Show Metadata"}
          >
            {showMetadata ? '🔧' : '⚙️'} 
          </IconButton>
        </Tabs>
      </Box>
      <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 0 }}> {/* Remove padding here, add in render functions */}
        {activeTabContent}
      </Box>
    </Paper>
  );
});

export default WordDetails;