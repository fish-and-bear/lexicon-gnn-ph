import React, { useCallback, useState, useEffect, useMemo } from 'react';
import { Definition, WordInfo, RelatedWord, NetworkLink, NetworkNode, WordForm, WordTemplate, Idiom, Affixation, DefinitionCategory, DefinitionLink, DefinitionRelation } from '../types';
// import { convertToBaybayin } from '../api/wordApi';
import './WordDetails.css';
// import './Tabs.css';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
// --- Commented out unused MUI imports ---
// import Tabs from '@mui/material/Tabs';
// import Tab from '@mui/material/Tab';
// import List from '@mui/material/List';
// import ListItem from '@mui/material/ListItem';
// import ListItemText from '@mui/material/ListItemText';
// import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
// import Accordion from '@mui/material/Accordion';
// import AccordionSummary from '@mui/material/AccordionSummary';
// import AccordionDetails from '@mui/material/AccordionDetails';
import IconButton from '@mui/material/IconButton';
// import CircularProgress from '@mui/material/CircularProgress';
// import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Link from '@mui/material/Link';
import { styled, useTheme, alpha } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery'; // Import useMediaQuery
// import Drawer from '@mui/material/Drawer';
// import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
// import RefreshIcon from '@mui/icons-material/Refresh';
// import Grid from '@mui/material/Grid';

// --- Commented out d3 --- 
// import * as d3 from 'd3';

// --- Commented out unused icons --- 
// const ExpandMoreIcon = () => <Typography sx={{ transform: 'rotate(90deg)', lineHeight: 0, color: 'text.secondary' }}>▶</Typography>;
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'primary.main' }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'error.main' }}>⏹️</Typography>;

interface WordDetailsProps {
  wordInfo: WordInfo;
  // --- Keep other props commented for now ---
  // etymologyTree: any;
  // isLoadingEtymology: boolean;
  // etymologyError: string | null;
  onWordLinkClick: (word: string) => void; // Keep as string for now
  // onEtymologyNodeClick: (node: any) => void;
}

// --- Remove unused helpers and styled components --- 
/*
interface AnyRecord { [key: string]: any; }
const formatRelationType = (type: string): string => { ... };
const relationColors: { [key: string]: string } = { ... };
const graphColors = { ... };
const isColorLight = (hexColor: string): boolean => { ... };
const StyledAccordion = styled(Accordion)(({ theme }) => ({ ... }));
const StyledAccordionSummary = styled(AccordionSummary)(({ theme }) => ({ ... }));
const StyledAccordionDetails = styled(AccordionDetails)(({ theme }) => ({ ... }));
const DefinitionCard = styled(Box)(({ theme }) => ({ ... }));
const DefinitionItem = styled(Box)(({ theme }) => ({ ... }));
const SourceTag = styled(Typography)(({ theme }) => ({ ... }));
*/

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  onWordLinkClick,
}) => {
  const theme = useTheme();
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md'));
  const isDarkMode = theme.palette.mode === 'dark';

  // --- Remove internal state and effects --- 
  /*
  const [tabValue, setTabValue] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<string>('definitions');
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [audioElement, setAudioElement] = useState<HTMLAudioElement | null>(null);
  useEffect(() => { ... }, [wordInfo]); // Audio setup effect
  const handleTabChange = (event: React.SyntheticEvent, newValue: string) => { ... };
  const playAudio = useCallback(() => { ... }, [audioElement, isAudioPlaying]);
  useEffect(() => { ... }, [availableTabs, activeTab, wordInfo?.id]); // Tab reset effect
  */
  
  // --- Keep only essential helpers for renderHeader --- 
  const playAudio = useCallback(() => {
      // Dummy implementation for now
      console.log("Play audio clicked (dummy)");
  }, []);
  const isAudioPlaying = false; // Dummy value
  const stopAudio = useCallback(() => { console.log("Stop audio (dummy)"); }, []); // Dummy

  const renderHeader = () => {
    // Simplified version - relies only on wordInfo, isDarkMode, playAudio, isAudioPlaying
    if (!wordInfo) return null;
    const ipaPronunciation = wordInfo.pronunciations?.find(p => p.type === 'IPA');
    const hasAudio = wordInfo.pronunciations?.some(p => p.type === 'audio' && p.value);
    const tags = wordInfo.tags ? wordInfo.tags.split(',').map(tag => tag.trim()).filter(Boolean) : [];
    const mainColor = isDarkMode ? theme.palette.primary.light : theme.palette.primary.main;
    const headerBgColor = isDarkMode ? alpha(graphColors.main, 0.15) : alpha(graphColors.main, 0.07); // graphColors might be undefined here - use fallback
    const headerTextColor = isDarkMode ? theme.palette.primary.contrastText : theme.palette.getContrastText(alpha(mainColor, 0.07));
    
    return (
      <Box sx={{ 
        bgcolor: isDarkMode ? 'rgba(30, 40, 60, 0.4)' : alpha(mainColor, 0.07), // Use mainColor directly
        color: headerTextColor, 
        pt: 3, pb: 3, pl: 3, pr: 2,
        borderBottom: isDarkMode ? '1px solid rgba(255,255,255,0.05)' : `1px solid ${alpha(theme.palette.divider, 0.08)}`,
      }}>
        <Stack direction="row" spacing={1} alignItems="flex-start" flexWrap="nowrap" sx={{ mb: 1.5, width: '100%' }}>
          <Typography 
            variant="h3" component="h1" 
            sx={{ flexGrow: 1, minWidth: 0, fontSize: { xs: '1.5rem', sm: '1.75rem', md: '2rem' }, fontWeight: 700 }}>
            {wordInfo.lemma}
          </Typography>
          {hasAudio && (
            <IconButton onClick={playAudio} size="medium" title={isAudioPlaying ? "Stop Audio" : "Play Audio"} sx={{ flexShrink: 0, color: mainColor, mt: 0.5 }}>
              {isAudioPlaying ? <StopCircleIcon /> : <VolumeUpIcon />}
            </IconButton>
          )}
        </Stack>
        {ipaPronunciation && (
          <Typography variant="h6" sx={{ fontStyle: 'italic', mb: 1.5, pl: 0.5, color: alpha(headerTextColor, 0.85) }}>
            /{ipaPronunciation.value}/
          </Typography>
        )}
        {/* Simplified tags rendering */}
        {tags.length > 0 && (
          <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 2 }}>
            {tags.map((tag) => (
              <Chip key={tag} label={tag} size="small" variant="outlined" />
            ))}
          </Stack>
        )}
      </Box>
    );
  };

  // --- Remove other render functions and hooks --- 
  /*
  const renderSafeHtml = ...
  const renderWordLink = ...
  const renderDefinitionsTab = ...
  const renderRelationsTab = ...
  const renderFormsAndTemplatesTab = ...
  const renderEtymologyTab = ...
  const renderSourcesInfoTab = ...
  const availableTabs = useMemo(...);
  */

  // --- Minimal Return --- 
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
        maxWidth: '100%'
      }}
    >
      {renderHeader()}
      {/* Placeholder for removed content */}
      <Box sx={{ p: 2, flexGrow: 1 }}>
        <Typography variant="h6">Word Details (Minimal Render)</Typography>
        <Typography>Only the header is currently rendered for debugging.</Typography>
      </Box>
    </Paper>
  );
});

export default WordDetails;