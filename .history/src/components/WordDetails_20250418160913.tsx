import React, { useCallback, useState, useEffect, useMemo } from 'react';
import { WordInfo, RelatedWord } from '../types'; // Keep only used types
import './WordDetails.css';

// MUI Imports - Minimal
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import IconButton from '@mui/material/IconButton';
import Link from '@mui/material/Link'; // Keep if used in renderHeader/renderWordLink
import { useTheme, alpha } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

// Minimal Icons
const VolumeUpIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'primary.main' }}>🔊</Typography>;
const StopCircleIcon = () => <Typography sx={{ fontSize: '1.2em', lineHeight: 0, color: 'error.main' }}>⏹️</Typography>;

interface WordDetailsProps {
  wordInfo: WordInfo;
  onWordLinkClick: (word: string) => void; // Keep this prop
  // Comment out others
  // etymologyTree: any;
  // isLoadingEtymology: boolean;
  // etymologyError: string | null;
  // onEtymologyNodeClick: (node: any) => void;
}

// No internal state, effects, or complex logic needed for this minimal version

const WordDetails: React.FC<WordDetailsProps> = React.memo(({
  wordInfo,
  onWordLinkClick,
}) => {
  const theme = useTheme();
  const isWideScreen = useMediaQuery(theme.breakpoints.up('md')); // Keep for potential layout adjustments
  const isDarkMode = theme.palette.mode === 'dark';

  // Dummy audio handlers for renderHeader if needed
  const playAudio = useCallback(() => {
      console.log("Play audio clicked (dummy)");
  }, []);
  const isAudioPlaying = false; // Dummy value

  // Keep renderHeader (or a simplified version)
  const renderHeader = () => {
    if (!wordInfo) return null;
    const ipaPronunciation = wordInfo.pronunciations?.find(p => p.type === 'IPA');
    const hasAudio = wordInfo.pronunciations?.some(p => p.type === 'audio' && p.value);
    const tags = wordInfo.tags ? wordInfo.tags.split(',').map(tag => tag.trim()).filter(Boolean) : [];
    const mainColor = isDarkMode ? theme.palette.primary.light : theme.palette.primary.main;
    const headerTextColor = isDarkMode ? theme.palette.primary.contrastText : theme.palette.getContrastText(alpha(mainColor, 0.07));

    return (
      <Box sx={{
        bgcolor: isDarkMode ? 'rgba(30, 40, 60, 0.4)' : alpha(mainColor, 0.07),
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

  // Minimal Return Statement
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