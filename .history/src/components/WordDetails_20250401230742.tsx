import React, { useCallback, useState, useEffect } from 'react';
import { WordInfo, RelatedWord } from '../types';
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

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  etymologyTree,
  isLoadingEtymology,
  etymologyError,
  onWordLinkClick,
  onEtymologyNodeClick
}) => {
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

    return (
      <Box sx={{ mb: 3, borderBottom: 1, borderColor: 'divider', pb: 2 }}>
        <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
           <Typography variant="h4" component="h1" sx={{ fontWeight: 500 }}>
             {wordInfo.lemma}
           </Typography>
           {hasAudio && (
             <Button onClick={playAudio} size="small" variant="outlined" sx={{ minWidth: 'auto', px: 1 }}>
               {isAudioPlaying ? 'Stop' : 'Play'}
             </Button>
           )}
        </Stack>
        {ipaPronunciation && (
            <Typography variant="subtitle1" color="text.secondary" sx={{ fontStyle: 'italic', mb: 1 }}>
              /{ipaPronunciation.value}/
            </Typography>
        )}
        {wordInfo.has_baybayin && wordInfo.baybayin_form && (
            <Typography variant="h5" sx={{ fontFamily: 'Noto Sans Baybayin, sans-serif', mb: 1 }}>
                {wordInfo.baybayin_form}
            </Typography>
            /* Optional: Add a "Baybayin Script" label if desired
            <Typography variant="caption" color="text.secondary" display="block">
                Baybayin Script
            </Typography>
            */
        )}
        {wordInfo.tags && (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 1 }}>
            {wordInfo.tags.split(',').map(tag => tag.trim()).filter(Boolean).map((tag) => (
              <Chip key={tag} label={tag} size="small" variant="outlined" />
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
      <List dense disablePadding>
        {wordInfo.definitions.map((def, index) => (
          <React.Fragment key={def.id || index}>
            <ListItem alignItems="flex-start">
              <ListItemText
                primary={
                   <Typography variant="body1">
                     {index + 1}. {def.definition}
                   </Typography>
                 }
                secondary={def.source ? `Source: ${def.source}` : null}
              />
            </ListItem>
            {index < wordInfo.definitions.length - 1 && <Divider component="li" variant="inset" />}
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
            <Accordion defaultExpanded={false} variant="outlined" sx={{ '&:before': { display: 'none' }, boxShadow: 'none', border: 'none', borderBottom: 1, borderColor: 'divider', '&:last-child': { borderBottom: 0 } }}>
                <AccordionSummary expandIcon={<Typography sx={{ fontSize: '0.8em' }}>▼</Typography>}>
                    <Typography variant="subtitle1">{title} ({relations.length})</Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ pt: 0 }}>
                    <Stack spacing={1.5}>
                        {sortedTypes.map((type) => (
                            <Box key={type}>
                                <Typography variant="overline" display="block" sx={{ lineHeight: 1.5, mb: 0.5 }}>{formatRelationType(type)}</Typography>
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
                                            />
                                        ) : null; // Skip rendering if lemma is missing
                                    })}
                                </Stack>
                            </Box>
                        ))}
                    </Stack>
                </AccordionDetails>
            </Accordion>
        );
    };

    return (
        <Box sx={{ mt: 1 }}>
            {renderRelationGroup("Derived From / Related To (Incoming)", wordInfo.incoming_relations, 'in')}
            {renderRelationGroup("Derives / Related From (Outgoing)", wordInfo.outgoing_relations, 'out')}
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
    if (!etymologyTree || (Array.isArray(etymologyTree) && etymologyTree.length === 0)) { // Adjust check based on actual structure
       return <Alert severity="info" sx={{ m: 2 }}>No etymology information available.</Alert>;
    }

    // Placeholder: Render raw data or a simple list
    // TODO: Implement a proper tree visualization component if needed
    return (
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>Etymology Tree</Typography>
        <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', fontSize: '12px', background: '#f5f5f5', padding: '10px', borderRadius: '4px' }}>
            {JSON.stringify(etymologyTree, null, 2)}
        </pre>
        {/* Implement interactive tree rendering here */}
      </Box>
    );
  };

  const renderCreditsTab = () => {
     if (!wordInfo?.sources || wordInfo.sources.length === 0) {
       return <Alert severity="info" sx={{ m: 2 }}>No source information available.</Alert>;
     }
     return (
       <List dense disablePadding>
         {wordInfo.sources.map((source, index) => (
           <ListItem key={index}>
             <ListItemText primary={source.name} secondary={source.url || 'No URL provided'} />
           </ListItem>
         ))}
       </List>
     );
  };

  // --- Main Component Return ---
  if (!wordInfo?.id) {
    // Render a placeholder or empty state if no word is selected
    return (
        <Paper elevation={2} sx={{ p: 3, height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Typography color="text.secondary">Select a word to see details.</Typography>
        </Paper>
    );
  }

  return (
    <Paper elevation={0} square sx={{ display: 'flex', flexDirection: 'column', height: '100%', borderLeft: 1, borderColor: 'divider' }}>
        <Box sx={{ p: 2 }}>
            {renderHeader()}
        </Box>

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs
              value={activeTab}
              onChange={handleTabChange}
              variant="scrollable"
              scrollButtons="auto"
              aria-label="Word details tabs"
              sx={{ minHeight: '48px' }} // Standard tab height
            >
              <Tab label="Definitions" value="definitions" sx={{ minWidth: 'auto', px: 2 }} />
              <Tab label="Relations" value="relations" sx={{ minWidth: 'auto', px: 2 }} />
              <Tab label="Etymology" value="etymology" sx={{ minWidth: 'auto', px: 2 }} />
              {/* <Tab label="Baybayin" value="baybayin" /> // Integrate into header? */}
              <Tab label="Sources" value="credits" sx={{ minWidth: 'auto', px: 2 }}/>
              {/* Add Metadata tab or button if needed */}
            </Tabs>
        </Box>

        <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 0 }}>
            {/* Render active tab content */}
            {activeTab === 'definitions' && renderDefinitionsTab()}
            {activeTab === 'relations' && renderRelationsTab()}
            {activeTab === 'etymology' && renderEtymologyTab()}
            {activeTab === 'credits' && renderCreditsTab()}
            {/* Add other tab content here */}
        </Box>

        {/* Optional: Metadata section can be placed here or triggered by a button */}
        {/* {renderMetadata(wordInfo)} */}
    </Paper>
  );
});

export default WordDetails;