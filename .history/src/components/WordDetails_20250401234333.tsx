import React, { useCallback, useState, useEffect } from 'react';
import { WordInfo, RelatedWord, Definition } from '../types';
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

// Styled components (optional, can also use sx prop)
const StyledAccordion = styled(Accordion)(({ theme }) => ({
  border: `1px solid ${theme.palette.divider}`,
  borderLeft: 0,
  borderRight: 0,
  boxShadow: 'none',
  '&:not(:last-child)': {
    borderBottom: 0,
  },
  '&:before': {
    display: 'none',
  },
  '&.Mui-expanded': {
    margin: 0, // Prevent margin shift on expand
  },
}));

const StyledAccordionSummary = styled(AccordionSummary)(({ theme }) => ({
  minHeight: 48, // Consistent height
  '& .MuiAccordionSummary-content': {
    margin: '12px 0', // Default Material UI spacing
  },
}));

const StyledAccordionDetails = styled(AccordionDetails)(({ theme }) => ({
  padding: theme.spacing(1, 2, 2), // top, horizontal, bottom
}));

const ExpandMoreIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0 }}>▼</Typography>; // Placeholder
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0 }}>🔊</Typography>; // Placeholder
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0 }}>⏹️</Typography>; // Placeholder

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

  const [activeTab, setActiveTab] = useState<string>('definitions'); // Use string for tab value
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);

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
    const ipaPronunciation = wordInfo.pronunciations?.find(p => p.type === 'ipa');
    const hasAudio = !!audioElement;
    const tags = wordInfo.tags ? wordInfo.tags.split(',').map(tag => tag.trim()).filter(Boolean) : [];

    // Use the graph's main color for header, adjust alpha
    const headerBgColor = isDarkMode
        ? alpha(graphColors.main, 0.6) // Slightly less transparent in dark mode
        : alpha(graphColors.main, 0.1); // Very light tint in light mode
    // Determine text color based on calculated header background
    const effectiveHeaderBg = theme.palette.augmentColor({ color: { main: headerBgColor } });
    const headerTextColor = effectiveHeaderBg.contrastText; // Use MUI contrast text

    return (
      <Box sx={{ bgcolor: headerBgColor, color: headerTextColor }}>
        <Box sx={{ p: 2.5 }}>
            {/* Lemma and Audio Button */}
            <Stack direction="row" spacing={1} alignItems="flex-start" sx={{ mb: 1 }}>
               <Typography variant="h3" component="h1" sx={{ fontWeight: 700, flexGrow: 1, lineHeight: 1.2 }}>
                 {wordInfo.lemma}
               </Typography>
               {hasAudio && (
                 <IconButton
                    onClick={playAudio}
                    size="medium"
                    title={isAudioPlaying ? "Stop Audio" : "Play Audio"}
                    // Ensure icon color contrasts with header BG
                    sx={{ color: headerTextColor, mt: 0.5, '&:hover': { bgcolor: alpha(headerTextColor, 0.1) } }}
                 >
                   {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
                 </IconButton>
               )}
            </Stack>

            {/* Pronunciation (IPA) */}
            {ipaPronunciation && (
                <Typography variant="h6" sx={{ color: alpha(headerTextColor, 0.85), fontStyle: 'italic', mb: 1.5, pl: 0.5 }}>
                  /{ipaPronunciation.value}/
                </Typography>
            )}

            {/* Baybayin */}
            {wordInfo.has_baybayin && wordInfo.baybayin_form && (
                 <Box sx={{ my: 2 }}>
                    <Typography variant="caption" sx={{ color: alpha(headerTextColor, 0.75), display: 'block', mb: 0.5 }}>
                        Baybayin Script
                    </Typography>
                    <Typography
                       variant="h4"
                       sx={{
                           fontFamily: 'Noto Sans Baybayin, sans-serif',
                           p: 1,
                           // Use a subtle inset background that works with headerTextColor
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
              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 2 }}>
                {tags.map((tag) => (
                  <Chip
                    key={tag}
                    label={tag}
                    size="small"
                    sx={{
                        color: alpha(headerTextColor, 0.9),
                        borderColor: alpha(headerTextColor, 0.5), // Slightly more visible border
                        bgcolor: 'transparent', // Transparent background
                        '& .MuiChip-label': { fontWeight: 500 }
                    }}
                    variant="outlined"
                  />
                ))}
              </Stack>
            )}
         </Box>
      </Box>
    );
  };

  const renderDefinitionsTab = () => {
    if (!wordInfo?.definitions || wordInfo.definitions.length === 0) {
      return <Alert severity="info" sx={{ m: 2 }}>No definitions available.</Alert>;
    }
    return (
      <List dense sx={{ py: 0 }}> {/* Remove default padding */}
        {wordInfo.definitions.map((def: Definition, index) => (
          <React.Fragment key={def.id || index}>
            <ListItem alignItems="flex-start" sx={{ py: 1.5 }}>
              <Typography variant="body1" sx={{ mr: 1.5, color: 'text.secondary' }}>{index + 1}.</Typography>
              <ListItemText
                primary={
                   <Typography variant="body1" component="span">
                     {def.text}
                   </Typography>
                 }
                secondary={
                  <Box component="span" sx={{ display: 'block', mt: 0.5 }}>
                    {def.sources && def.sources.length > 0 && (
                       <Typography variant="caption" color="text.secondary" component="span">
                           Sources: {def.sources.join(', ')}
                       </Typography>
                    )}
                    {/* TODO: Add examples if available in def.examples */}
                  </Box>
                 }
                sx={{ my: 0 }} // Remove default margins
              />
            </ListItem>
            {index < (wordInfo.definitions?.length ?? 0) - 1 && <Divider component="li" variant="inset" />}
          </React.Fragment>
        ))}
      </List>
    );
  };

  const renderRelationsTab = () => {
    const hasOutgoing = wordInfo.outgoing_relations && wordInfo.outgoing_relations.length > 0;
    const hasIncoming = wordInfo.incoming_relations && wordInfo.incoming_relations.length > 0;

    if (!hasOutgoing && !hasIncoming) {
      return <Alert severity="info" sx={{ m: 2 }}>No relationship information available.</Alert>;
    }

    const renderRelationGroup = (title: string, relations: WordInfo['outgoing_relations'] | WordInfo['incoming_relations'], direction: 'in' | 'out') => {
        if (!relations || relations.length === 0) return null;

        const grouped: { [key: string]: typeof relations } = {};
        relations.forEach(rel => {
            const type = rel.relation_type || 'unknown';
            if (!grouped[type]) grouped[type] = [];
            grouped[type].push(rel);
        });
        const sortedTypes = Object.keys(grouped).sort((a, b) => a.localeCompare(b));

        return (
            <StyledAccordion defaultExpanded={false} disableGutters>
                <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>{title}</Typography>
                    <Chip label={relations.length} size="small" sx={{ ml: 1.5 }}/>
                </StyledAccordionSummary>
                <StyledAccordionDetails>
                    <Stack spacing={2}>
                        {sortedTypes.map((type) => {
                             // Get the corresponding graph color, fallback to default
                             const chipColorHex = (graphColors as Record<string, string>)[type.toLowerCase()] || graphColors.default;
                             const chipTextColor = isColorLight(chipColorHex) ? theme.palette.common.black : theme.palette.common.white;
                            return (
                                <Box key={type}>
                                    <Typography variant="overline" display="block" sx={{ lineHeight: 1.5, mb: 0.5, color: 'text.secondary' }}>{formatRelationType(type)}</Typography>
                                    <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap">
                                        {grouped[type].map((relatedItem, index) => {
                                            const relatedWord: RelatedWord | undefined = direction === 'out' ? relatedItem.target_word : relatedItem.source_word;
                                            return relatedWord?.lemma ? (
                                                <Chip
                                                    key={`${type}-${index}-${relatedWord.id}`}
                                                    label={relatedWord.lemma}
                                                    onClick={() => onWordLinkClick(relatedWord.lemma)}
                                                    clickable
                                                    size="small"
                                                    // Use direct hex colors ensuring contrast
                                                    sx={{
                                                        bgcolor: chipColorHex,
                                                        color: chipTextColor,
                                                        cursor: 'pointer',
                                                        '& .MuiChip-label': { fontWeight: 500 },
                                                        '&:hover': { bgcolor: alpha(chipColorHex, 0.85) }
                                                    }}
                                                    title={`View ${relatedWord.lemma}`}
                                                />
                                            ) : null;
                                        })}
                                    </Stack>
                                </Box>
                            );
                        })}
                    </Stack>
                </StyledAccordionDetails>
            </StyledAccordion>
        );
    };

    return (
        <Box sx={{ width: '100%' }}>
            {renderRelationGroup("Incoming Relations", wordInfo.incoming_relations, 'in')}
            {renderRelationGroup("Outgoing Relations", wordInfo.outgoing_relations, 'out')}
        </Box>
    );
  };

  const renderEtymologyTab = () => {
    if (isLoadingEtymology) {
      return <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}><CircularProgress /></Box>;
    }
    if (etymologyError) {
      return <Alert severity="error" sx={{ m: 2 }}>{etymologyError}</Alert>;
    }
    // Check if etymologyTree exists and has nodes
    if (!etymologyTree || !etymologyTree.nodes || etymologyTree.nodes.length === 0) {
       return <Alert severity="info" sx={{ m: 2 }}>No etymology information available.</Alert>;
    }

    // Assume Etymology Tree structure based on usage
    type EtymologyNode = { id: number; label: string; language?: string; [key: string]: any };
    type EtymologyEdge = { source: number; target: number; [key: string]: any };
    type EtymologyTreeMap = { [id: number]: EtymologyNode };

    // Basic List Rendering of Etymology Nodes (Improved from JSON)
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
                    <Typography variant="body2" component="span" sx={{ fontWeight: level === 0 ? 600 : 400 }}>
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
      <Box sx={{ p: 0 }}>
          <List dense sx={{ pt: 1 }}>
             {rootIds.length > 0
                ? rootIds.map((rootId: number) => renderNode(rootId, nodeMap, etymologyTree.edges))
                : <ListItem><Alert severity="warning" variant="outlined" sx={{ width: '100%' }}>Could not determine root etymology node.</Alert></ListItem> }
          </List>
      </Box>
    );
  };

  const renderCreditsTab = () => {
     if (!wordInfo?.credits || wordInfo.credits.length === 0) {
       return <Alert severity="info" sx={{ m: 2 }}>No source information available.</Alert>;
     }
     return (
       <List dense sx={{ py: 1 }}>
         {wordInfo.credits.map((credit, index) => (
           <ListItem key={credit.id || index} sx={{ py: 1 }}>
             <ListItemText
                primary={credit.credit}
                // Add secondary info if relevant, e.g., URLs, roles
             />
           </ListItem>
         ))}
       </List>
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
               <Box sx={{ flexGrow: 1, overflowY: 'auto', position: 'relative', bgcolor: 'background.default' }}>
                  <Box sx={{ p: { xs: 1.5, sm: 2, md: 2.5 } }}>
                      {activeTab === 'definitions' && renderDefinitionsTab()}
                      {activeTab === 'relations' && renderRelationsTab()}
                      {activeTab === 'etymology' && renderEtymologyTab()}
                      {activeTab === 'credits' && renderCreditsTab()}
                  </Box>
               </Box>
            );

            const tabs = (
               <Tabs
                  orientation={isWideScreen ? "vertical" : "horizontal"}
                  variant="scrollable"
                  scrollButtons="auto"
                  allowScrollButtonsMobile
                  value={activeTab}
                  onChange={handleTabChange}
                  aria-label="Word details sections"
                  // Use main graph color for indicator, secondary for text
                  textColor="secondary"
                  indicatorColor="primary" // Set indicator to primary
                  TabIndicatorProps={{
                     style: {
                        backgroundColor: graphColors.main // Use direct color for indicator
                     }
                  }}
                  sx={isWideScreen ? {
                      borderRight: 1, borderColor: 'divider', minWidth: 180,
                      bgcolor: 'background.paper', pt: 1,
                      '& .MuiTab-root': { alignItems: 'flex-start', textAlign: 'left', minHeight: 48, textTransform: 'none', fontWeight: 500, pt: 1.5, pb: 1.5, pl: 2.5 },
                       // Use primary color for selected vertical tab text to match indicator
                       '& .Mui-selected': { color: `${graphColors.main} !important` },
                      '& .MuiTabs-indicator': { left: 0, width: 4, borderRadius: 1 }
                  } : {
                      borderBottom: 1, borderColor: 'divider', minHeight: 48,
                      bgcolor: 'background.paper',
                      '& .MuiTab-root': { textTransform: 'none', minWidth: 'auto', px: 2 },
                       // Use primary color for selected horizontal tab text
                       '& .Mui-selected': { color: `${graphColors.main} !important` },
                  }}
                >
                  <Tab label="Definitions" value="definitions" />
                  <Tab label="Relations" value="relations" />
                  <Tab label="Etymology" value="etymology" />
                  <Tab label="Sources" value="credits" />
                </Tabs>
            );

            return (
    <Paper elevation={2} square sx={{ display: 'flex', flexDirection: 'column', height: '100%', bgcolor: 'background.paper', overflow: 'hidden' }}>

      {/* Conditional Layout based on screen size */}
      {isWideScreen ? (
          <>
            {renderHeader()}
            <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'row', overflow: 'hidden' }}>
               {tabs}
               {tabContent}
            </Box>
          </>
      ) : (
          <>
            {renderHeader()}
            {tabs}
            {tabContent}
          </>
      )}
      {/* Add comment about resizing parent */}
      {/* For resizable sidebar functionality, the parent component (e.g., WordExplorer) needs layout adjustments (e.g., using SplitPane) */}
    </Paper>
  );
});

export default WordDetails;