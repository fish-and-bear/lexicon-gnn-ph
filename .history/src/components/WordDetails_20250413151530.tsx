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
      return <Alert severity="info">No definitions available for this word.</Alert>;
    }
    return (
      // Use theme spacing
      <List disablePadding sx={{ pt: theme.spacing(1) }}>
        {wordInfo.definitions.map((def: Definition, index: number) => (
          <React.Fragment key={index}>
            <ListItem alignItems="flex-start" sx={{ flexDirection: 'column', gap: 0.5, pb: theme.spacing(1.5), pt: theme.spacing(1.5) }}>
              <ListItemText
                primaryTypographyProps={{ variant: 'body1', fontWeight: 500 }}
                primary={def.text}
              />
              {def.examples && def.examples.length > 0 && (
                <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', pl: theme.spacing(0.5) }}>
                  Example: "{def.examples[0]}"
                </Typography>
              )}
              {def.part_of_speech && (
                 <Chip label={def.part_of_speech?.name_en ?? 'Unknown POS'} size="small" variant="outlined" sx={{ mt: 1 }}/>
              )}
            </ListItem>
            {index < wordInfo.definitions!.length - 1 && <Divider component="li" />}
          </React.Fragment>
        ))}
      </List>
    );
  };

  const renderRelationsTab = () => {
    const hasIncoming = wordInfo.incoming_relations && wordInfo.incoming_relations.length > 0;
    const hasOutgoing = wordInfo.outgoing_relations && wordInfo.outgoing_relations.length > 0;

    // Enhanced debug logging
    console.log("WordDetails renderRelationsTab - wordInfo:", {
      id: wordInfo.id,
      lemma: wordInfo.lemma,
      hasIncoming,
      hasOutgoing,
      incomingCount: wordInfo.incoming_relations?.length || 0,
      outgoingCount: wordInfo.outgoing_relations?.length || 0,
      incomingSample: hasIncoming ? wordInfo.incoming_relations?.slice(0, 2) : null,
      outgoingSample: hasOutgoing ? wordInfo.outgoing_relations?.slice(0, 2) : null
    });

    if (!hasIncoming && !hasOutgoing) {
        return <Alert severity="info">No relations available for this word.</Alert>;
    }

    // Debug relation data
    console.log("Rendering relations tab with data:", {
      incoming: wordInfo.incoming_relations,
      outgoing: wordInfo.outgoing_relations
    });

    const renderRelationGroup = (title: string, relations: WordInfo['outgoing_relations'] | WordInfo['incoming_relations'], direction: 'in' | 'out') => {
      if (!relations || relations.length === 0) return null;

      // Check if relations have the expected structure
      const hasValidStructure = relations.every(rel => {
        const wordObj = direction === 'in' ? rel.source_word : rel.target_word;
        
        if (!wordObj) {
          console.warn(`Relation missing ${direction === 'in' ? 'source' : 'target'} word:`, rel);
          return false;
        }
        
        if (!rel.relation_type) {
          console.warn("Relation missing relation_type:", rel);
          return false;
        }
        
        return true;
      });

      if (!hasValidStructure) {
        console.error(`Invalid ${direction === 'in' ? 'incoming' : 'outgoing'} relations structure`);
        return (
          <StyledAccordion defaultExpanded={false} disableGutters>
            <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>{title}</Typography>
              <Chip label={relations.length} size="small" sx={{ ml: 1.5 }}/>
            </StyledAccordionSummary>
            <StyledAccordionDetails>
              <Alert severity="warning">
                Relations data is incomplete or has an unexpected structure.
              </Alert>
            </StyledAccordionDetails>
          </StyledAccordion>
        );
      }

      // Group relations by type
      const grouped = relations.reduce((acc, item) => {
        // Ensure relation_type exists and is a string
        const type = item.relation_type || 'unknown';
        if (!acc[type]) acc[type] = [];
        acc[type].push(item);
        return acc;
      }, {} as Record<string, typeof relations>);

      const sortedTypes = Object.keys(grouped).sort();

      return (
        <StyledAccordion defaultExpanded={false} disableGutters>
          <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>{title}</Typography>
            <Chip label={relations.length} size="small" sx={{ ml: 1.5 }}/>
          </StyledAccordionSummary>
          <StyledAccordionDetails>
            {/* Use theme spacing for gap */}
            <Stack spacing={theme.spacing(2.5)}>
              {sortedTypes.map((type) => {
                const chipColorHex = (graphColors as Record<string, string>)[type.toLowerCase()] || graphColors.default;
                // For outlined, we want the border and text to use the color
                const chipTextColor = chipColorHex; // Use hex directly for text/border
                const isLightBg = theme.palette.mode === 'light';

                return (
                  <Box key={type}>
                    <Typography variant="overline" display="block" sx={{ lineHeight: 1.5, mb: 0.5, color: 'text.secondary' }}>{formatRelationType(type)}</Typography>
                     {/* Use theme spacing for chip gap */}
                    <Stack direction="row" spacing={theme.spacing(1)} useFlexGap flexWrap="wrap">
                      {grouped[type].map((relatedItem, index) => {
                        // Select the appropriate word object based on direction
                        const relatedWord: RelatedWord | undefined = direction === 'out' ? relatedItem.target_word : relatedItem.source_word;
                        
                        // Skip rendering if related word is undefined or missing lemma
                        if (!relatedWord || !relatedWord.lemma) {
                          console.warn(`Missing ${direction === 'out' ? 'target' : 'source'} word in relation:`, relatedItem);
                          return null;
                        }
                        
                        return (
                          <Chip
                            key={`${type}-${index}-${relatedWord.id}`}
                            label={relatedWord.lemma}
                            onClick={() => onWordLinkClick(relatedWord.lemma)}
                            clickable
                            size="small"
                            variant="outlined" // Use outlined variant
                            sx={{
                              color: chipTextColor,
                              borderColor: chipTextColor,
                              cursor: 'pointer',
                              '& .MuiChip-label': { fontWeight: 500 },
                              // More subtle hover: slight background fill
                              '&:hover': {
                                  bgcolor: alpha(chipTextColor, isLightBg ? 0.1 : 0.15),
                              },
                              height: 'auto', // Allow chip height to adjust
                              padding: theme.spacing(0.25, 0.75)
                            }}
                            title={`View ${relatedWord.lemma}`}
                          />
                        );
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
    
    // Handle case where there's no etymology data from either source
    if (!hasWordEtymologies && !hasEtymologyTreeData) {
      return <Alert severity="info" sx={{ m: 2 }}>No etymology information available.</Alert>;
    }
    
    // If there's etymology data in the word itself, display it regardless of tree
    if (hasWordEtymologies) {
      console.log("Rendering etymology from word data:", wordInfo.etymologies);
      
      return (
        <Box sx={{ p: 2 }}>
          {/* Display direct etymology data from word */}
          <List dense>
            {wordInfo.etymologies!.map((etym, index) => (
              <ListItem key={index} sx={{ 
                display: 'block', 
                py: 1.5,
                borderBottom: index < wordInfo.etymologies!.length - 1 ? 
                  `1px solid ${theme.palette.divider}` : 'none'
              }}>
                <ListItemText
                  primary={
                    <Typography variant="body1" sx={{ fontWeight: 500, mb: 0.5 }}>
                      Etymology {index + 1}
                    </Typography>
                  }
                  secondary={
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                      {/* Main text */}
                      <Typography variant="body2" component="div" sx={{ whiteSpace: 'pre-wrap' }}>
                        {etym.text || etym.etymology_text}
                      </Typography>
                      
                      {/* Languages */}
                      {etym.languages && etym.languages.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" component="div" color="text.secondary" sx={{ mb: 0.5 }}>
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
                                  height: 24
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Components */}
                      {etym.components && etym.components.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" component="div" color="text.secondary" sx={{ mb: 0.5 }}>
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
                                  height: 24
                                }}
                              />
                            ))}
                          </Stack>
                        </Box>
                      )}
                      
                      {/* Sources */}
                      {etym.sources && etym.sources.length > 0 && (
                        <Typography variant="caption" component="div" color="text.secondary" sx={{ mt: 1 }}>
                          Sources: {etym.sources.join(', ')}
                        </Typography>
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
              <Typography variant="subtitle2" sx={{ mb: 1.5 }}>
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
                  <Box sx={{ p: theme.spacing(3) }}>
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
                      borderRight: 1, borderColor: 'divider', minWidth: 240,
                      bgcolor: 'background.paper', pt: theme.spacing(1),
                      '& .MuiTab-root': { alignItems: 'flex-start', textAlign: 'left', minHeight: 48, textTransform: 'none', fontWeight: 500, pt: theme.spacing(1.5), pb: theme.spacing(1.5), pl: theme.spacing(2.5) },
                       // Use primary color for selected vertical tab text to match indicator
                       '& .Mui-selected': { color: `${graphColors.main} !important` },
                      '& .MuiTabs-indicator': { left: 0, width: 4, borderRadius: 1 }
                  } : {
                      borderBottom: 1, borderColor: 'divider', minHeight: 48,
                      bgcolor: 'background.paper',
                      '& .MuiTab-root': { textTransform: 'none', minWidth: 'auto', px: theme.spacing(2.5) },
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