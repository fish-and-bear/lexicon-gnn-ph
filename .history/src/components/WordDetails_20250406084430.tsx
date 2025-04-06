import React, { useCallback, useState, useEffect } from 'react';
import {
  WordInfo,
  RelatedWord,
  Definition, // Cleaned type
  Etymology, // Cleaned type
  Pronunciation, // Cleaned type (includes raw value)
  Affixation, // Raw type
  WordForm, // Raw type
  WordTemplate, // Raw type
  Relation, // Raw type
  Credit, // Raw type
  BasicWord, // Added for Relation word nesting
} from '../types';
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

// Helper function to safely parse JSON strings or comma-separated strings into arrays/objects
// Handles null/undefined/empty strings gracefully.
const safeParse = (input: string | Record<string, any> | null | undefined, returnType: 'array' | 'object' = 'array'): any[] | Record<string, any> => {
  if (typeof input === 'object' && input !== null) {
    // If it's already an object, return it directly if object is expected
    // Ensure we return a copy for objects to avoid mutation issues if needed
    return returnType === 'object' ? { ...input } : {};
  }
  if (!input || typeof input !== 'string') {
    return returnType === 'array' ? [] : {};
  }
  try {
    // First, try parsing as JSON
    const parsed = JSON.parse(input);
    if (returnType === 'array' && Array.isArray(parsed)) {
      return parsed;
    } else if (returnType === 'object' && typeof parsed === 'object' && !Array.isArray(parsed) && parsed !== null) {
      return parsed;
    } else if (returnType === 'array') {
      // If JSON parse didn't yield an array, treat as single element array
      return [parsed];
    }
    // If expected object but got array/primitive, return empty object
    return {};

  } catch (e) {
    // If JSON parsing fails, assume comma-separated string for arrays
    if (returnType === 'array') {
        // Handle cases like just "tag1" without commas
        return input.split(',').map(s => s.trim()).filter(Boolean);
    }
    // Cannot parse as object if not valid JSON
    return {};
  }
};

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
    // Use the cleaned Pronunciation type from WordInfo
    const ipaPronunciation = wordInfo.pronunciations?.find(p => p.type === 'ipa' || p.type === 'IPA');
    const audioPronunciation = wordInfo.pronunciations?.find(p => p.type === 'audio' && p.value);
    const hasAudio = !!audioPronunciation;
    // Derive tagsArray from wordInfo.tags string
    const tagsArray = wordInfo.tags && typeof wordInfo.tags === 'string' 
                      ? wordInfo.tags.split(',').map(t => t.trim()).filter(Boolean)
                      : [];

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
          {/* Display Language */}
          {wordInfo.language && (
            <Chip
               label={wordInfo.language.name_en || wordInfo.language.code}
               size="small"
               variant="outlined"
               sx={{ ml: 2, mt: 0.5, color: alpha(headerTextColor, 0.8), borderColor: alpha(headerTextColor, 0.4) }}
            />
          )}
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
            /{ipaPronunciation.value}/ {/* Use value directly from Pronunciation type */}
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

        {/* Tags - Use tagsArray */}
        {tagsArray.length > 0 && ( // Check the derived array
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: theme.spacing(2) }}>
            {tagsArray.map((tag) => ( // Map over the derived array
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
        {wordInfo.definitions.map((def: Definition, index: number) => {
            // Examples are already string[] in the cleaned Definition type
            const examples = def.examples;

            return (
              <React.Fragment key={def.id || index}>
                <ListItem alignItems="flex-start" sx={{ flexDirection: 'column', gap: 0.5, pb: theme.spacing(1.5), pt: theme.spacing(1.5) }}>
                  <ListItemText
                    primaryTypographyProps={{ variant: 'body1', fontWeight: 500 }}
                    primary={def.definition_text} // Use definition_text
                  />
                  {examples && examples.length > 0 && (
                    <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic', pl: theme.spacing(0.5) }}>
                      Example: "{examples[0]}" {/* Access first example */}
                    </Typography>
                  )}
                  {def.part_of_speech && (
                     <Chip label={def.part_of_speech?.name_en ?? 'Unknown POS'} size="small" variant="outlined" sx={{ mt: 1 }}/>
                  )}
                </ListItem>
                {index < wordInfo.definitions!.length - 1 && <Divider component="li" />}
              </React.Fragment>
            );
        })}
      </List>
    );
  };

  const renderRelationsTab = () => {
    const hasIncoming = wordInfo.incoming_relations && wordInfo.incoming_relations.length > 0;
    const hasOutgoing = wordInfo.outgoing_relations && wordInfo.outgoing_relations.length > 0;

    if (!hasIncoming && !hasOutgoing) {
        return <Alert severity="info">No relations available for this word.</Alert>;
    }

    const renderRelationGroup = (title: string, relations: WordInfo['outgoing_relations'] | WordInfo['incoming_relations'], direction: 'in' | 'out') => {
      if (!relations || relations.length === 0) return null;

      // Group relations by type
      const grouped = relations.reduce((acc, item) => {
        const type = item.relation_type;
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
                      {grouped[type].map((relatedItem: Relation, index) => { // Use Relation type
                        // Access nested source/target word correctly
                        const relatedWord: BasicWord | undefined = direction === 'out' ? relatedItem.target_word : relatedItem.source_word;
                        return relatedWord?.lemma ? (
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

  const renderAffixationsTab = () => {
    const hasRootAffix = wordInfo.root_affixations && wordInfo.root_affixations.length > 0;
    const hasAffixedAffix = wordInfo.affixed_affixations && wordInfo.affixed_affixations.length > 0;
    const rootWord = wordInfo.root_word;

    if (!hasRootAffix && !hasAffixedAffix && !rootWord) {
      return <Alert severity="info">No affixation or root word information available.</Alert>;
    }

    const renderAffixGroup = (title: string, affixations: Affixation[] | null | undefined, direction: 'root' | 'affixed') => {
      if (!affixations || affixations.length === 0) return null;

      return (
        <Box mb={3}>
          <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 500 }}>{title}</Typography>
          <List dense disablePadding>
            {affixations.map((affix: Affixation, index: number) => {
              const relatedWord = direction === 'root' ? affix.affixed_word : affix.root_word;
              const affixType = affix.affix_type;
              return (
                <ListItem key={affix.id || index}>
                  <ListItemText
                    primary={relatedWord ? (
                        <Link component="button" variant="body2" onClick={() => onWordLinkClick(relatedWord.lemma)} sx={{ mr: 1 }}>
                            {relatedWord.lemma}
                        </Link>
                    ) : "Unknown Word"}
                    secondary={`via ${affixType}`}
                  />
                </ListItem>
              );
            })}
          </List>
        </Box>
      );
    };

    return (
      <Box sx={{ p: theme.spacing(2) }}>
        {rootWord && (
           <Box mb={3}>
             <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 500 }}>Root Word</Typography>
             <Chip
                key={rootWord.id}
                label={rootWord.lemma}
                onClick={() => onWordLinkClick(rootWord.lemma)}
                clickable
                size="small"
                variant="outlined"
                sx={{ cursor: 'pointer', '&:hover': { bgcolor: alpha(theme.palette.text.primary, 0.08) } }}
              />
           </Box>
        )}
        {renderAffixGroup("Derived from this word (as root)", wordInfo.root_affixations, 'root')}
        {renderAffixGroup("Formed from affixation (as result)", wordInfo.affixed_affixations, 'affixed')}
      </Box>
    );
  };

  const renderWordFormsTab = () => {
    if (!wordInfo.forms || wordInfo.forms.length === 0) {
        return <Alert severity="info">No alternative forms available.</Alert>;
    }
    return (
        <List dense sx={{ py: 1 }}>
            {wordInfo.forms.map((form: WordForm, index: number) => {
                // Safely parse the tags field which might be a JSON string or an object already
                const tags = safeParse(form.tags, 'object') as Record<string, any>;
                return (
                    <ListItem key={form.id || index} sx={{ py: 1 }}>
                        <ListItemText
                            primary={form.form}
                            secondary={form.is_canonical ? 'Canonical' : form.is_primary ? 'Primary' : null}
                        />
                        {Object.entries(tags).length > 0 && (
                            <Stack direction="row" spacing={0.5} ml={2} useFlexGap flexWrap="wrap">
                                {Object.entries(tags).map(([key, value]) => (
                                    <Chip key={key} label={`${key}: ${value}`} size="small" variant="outlined" />
                                ))}
                            </Stack>
                        )}
                    </ListItem>
                );
            })}
        </List>
    );
  };

  const renderWordTemplatesTab = () => {
      if (!wordInfo.templates || wordInfo.templates.length === 0) {
          return <Alert severity="info">No word templates available.</Alert>;
      }
      return (
          <List dense sx={{ py: 1 }}>
              {wordInfo.templates.map((template: WordTemplate, index: number) => {
                  // Safely parse the args field
                  const args = safeParse(template.args, 'object') as Record<string, any>;
                  return (
                      <ListItem key={template.id || index} sx={{ py: 1 }}>
                          <ListItemText
                              primary={template.template_name}
                              secondary={`Expansion: ${template.expansion || 'N/A'}`}
                          />
                          {Object.entries(args).length > 0 && (
                             <Stack direction="row" spacing={0.5} ml={2} useFlexGap flexWrap="wrap">
                                {Object.entries(args).map(([key, value]) => (
                                    <Chip key={key} label={`${key}: ${value}`} size="small" variant="outlined" />
                                ))}
                             </Stack>
                          )}
                      </ListItem>
                  );
              })}
          </List>
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
                  <Box sx={{ p: theme.spacing(3) }}>
                      {activeTab === 'definitions' && renderDefinitionsTab()}
                      {activeTab === 'relations' && renderRelationsTab()}
                      {activeTab === 'affixations' && renderAffixationsTab()}
                      {activeTab === 'forms' && renderWordFormsTab()}
                      {activeTab === 'templates' && renderWordTemplatesTab()}
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
                  <Tab label="Affixations" value="affixations" />
                  <Tab label="Forms" value="forms" />
                  <Tab label="Templates" value="templates" />
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