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
import { styled, useTheme } from '@mui/material/styles';

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
  console.log("WordDetails rendering with wordInfo:", wordInfo); // Keep debug log

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

    return (
      <Box sx={{ mb: 2, pb: 2 }}>
        {/* Lemma and Audio Button */}
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
           <Typography variant="h4" component="h1" sx={{ fontWeight: 600, flexGrow: 1 }}>
             {wordInfo.lemma}
           </Typography>
           {hasAudio && (
             <IconButton onClick={playAudio} size="medium" title={isAudioPlaying ? "Stop Audio" : "Play Audio"} color="primary">
               {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
             </IconButton>
           )}
        </Stack>

        {/* Pronunciation (IPA) */}
        {ipaPronunciation && (
            <Typography variant="body1" color="text.secondary" sx={{ fontStyle: 'italic', mb: 1.5 }}>
              /{ipaPronunciation.value}/
            </Typography>
        )}

        {/* Baybayin */}
        {wordInfo.has_baybayin && wordInfo.baybayin_form && (
             <Box sx={{ mb: 1.5 }}>
                <Typography
                   variant="h5"
                   sx={{
                       fontFamily: 'Noto Sans Baybayin, sans-serif',
                       p: 1,
                       bgcolor: 'action.hover',
                       borderRadius: 1,
                       display: 'inline-block'
                   }}
                >
                    {wordInfo.baybayin_form}
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                    Baybayin Script
                </Typography>
            </Box>
        )}

        {/* Tags */}
        {tags.length > 0 && (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 1.5 }}>
            {tags.map((tag) => (
              <Chip key={tag} label={tag} size="small" />
            ))}
          </Stack>
        )}
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
            // Use styled components for cleaner look
            <StyledAccordion defaultExpanded={false} disableGutters>
                <StyledAccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 500 }}>{title}</Typography>
                    <Chip label={relations.length} size="small" sx={{ ml: 1.5 }}/>
                </StyledAccordionSummary>
                <StyledAccordionDetails>
                    <Stack spacing={2}> {/* Increased spacing between types */}
                        {sortedTypes.map((type) => (
                            <Box key={type}>
                                <Typography variant="overline" display="block" sx={{ lineHeight: 1.5, mb: 0.5, color: 'text.secondary' }}>
                                    {formatRelationType(type)}
                                </Typography>
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
                                                variant="outlined"
                                                title={`View ${relatedWord.lemma}`}
                                                sx={{ 
                                                    cursor: 'pointer', 
                                                    '&:hover': { 
                                                        backgroundColor: 'action.hover' 
                                                    }
                                                }} 
                                            />
                                        ) : null; // Skip rendering if lemma is missing
                                    })}
                                </Stack>
                            </Box>
                        ))}
                    </Stack>
                </StyledAccordionDetails>
            </StyledAccordion>
        );
    };

    return (
        <Box sx={{ width: '100%' }}> {/* Ensure Box takes full width */}
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

    // Basic List Rendering of Etymology Nodes (Improved from JSON)
    const renderNode = (nodeId: number, nodes: { [id: number]: any }, edges: any[], level = 0) => {
       const node = nodes[nodeId];
       if (!node) return null;

       // Find children (nodes where the current node is a source)
       const childrenEdges = edges.filter(edge => edge.source === nodeId);
       const childrenIds = childrenEdges.map(edge => edge.target);

       return (
          <ListItem key={node.id} sx={{ pl: level * 2.5, display: 'block' }}> {/* Indentation based on level */}
             <ListItemText
                primary={
                    <Typography variant="body2" component="span" sx={{ fontWeight: level === 0 ? 600 : 400 }}>
                        {node.label}
                    </Typography>
                }
                secondary={node.language ? `(${node.language})` : null}
                sx={{ my: 0.5 }}
             />
             {childrenIds.length > 0 && (
                <List dense disablePadding sx={{ pl: 1.5 }}>
                   {childrenIds.map(childId => renderNode(childId, nodes, edges, level + 1))}
                </List>
             )}
          </ListItem>
       );
    };

    // Find the root node(s) - nodes that are not targets of any edge
    const targetIds = new Set(etymologyTree.edges.map(edge => edge.target));
    const rootIds = etymologyTree.nodes
                      .filter(node => !targetIds.has(node.id))
                      .map(node => node.id);

    // Build a map for quick node lookup
    const nodeMap = etymologyTree.nodes.reduce((acc, node) => {
        acc[node.id] = node;
        return acc;
    }, {} as { [id: number]: any });

    return (
      <Box sx={{ p: 0 }}>
          <List dense sx={{ pt: 1 }}>
             {rootIds.length > 0 
                ? rootIds.map(rootId => renderNode(rootId, nodeMap, etymologyTree.edges))
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
       <List dense sx={{ py: 0 }}>
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
        <Paper elevation={2} sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', bgcolor: 'background.default' }}>
            <Typography color="text.secondary">Select a word to see details.</Typography>
        </Paper>
              );
            }
            
            return (
    <Paper elevation={0} square sx={{ display: 'flex', flexDirection: 'column', height: '100%', bgcolor: 'background.paper' }}>
        {/* Header Area */}
        <Box sx={{ p: 2.5 }}>
            {renderHeader()}
        </Box>
        <Divider />

        {/* Tabs Area */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              allowScrollButtonsMobile
              aria-label="Word details sections"
              sx={{ minHeight: 48 }}
            >
              <Tab label="Definitions" value="definitions" sx={{ textTransform: 'none', minWidth: 'auto', px: 2 }} />
              <Tab label="Relations" value="relations" sx={{ textTransform: 'none', minWidth: 'auto', px: 2 }} />
              <Tab label="Etymology" value="etymology" sx={{ textTransform: 'none', minWidth: 'auto', px: 2 }} />
              <Tab label="Sources" value="credits" sx={{ textTransform: 'none', minWidth: 'auto', px: 2 }}/>
            </Tabs>
        </Box>

        {/* Tab Content Area */}
        <Box sx={{ flexGrow: 1, overflowY: 'auto', position: 'relative' /* For potential absolute positioning inside */ }}>
            {/* Add subtle padding within the scrollable area */}
            <Box sx={{ p: { xs: 1.5, sm: 2 } }}>
              {activeTab === 'definitions' && renderDefinitionsTab()}
              {activeTab === 'relations' && renderRelationsTab()}
              {activeTab === 'etymology' && renderEtymologyTab()}
              {activeTab === 'credits' && renderCreditsTab()}
            </Box>
        </Box>

    </Paper>
  );
});

export default WordDetails;