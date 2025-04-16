import React, { useEffect, useState } from 'react';
import { getBaybayinStatistics } from '../api/wordApi';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import { styled } from '@mui/material/styles';
import Tooltip from '@mui/material/Tooltip';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import LinearProgress from '@mui/material/LinearProgress';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  backgroundColor: '#f8f9fa',
  border: '1px solid rgba(0, 0, 0, 0.1)',
}));

const StatCard = styled(Card)(({ theme }) => ({
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  backgroundColor: 'white',
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  transition: 'transform 0.3s, box-shadow 0.3s',
  '&:hover': {
    transform: 'translateY(-5px)',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)',
  },
}));

const StatValue = styled(Typography)(({ theme }) => ({
  fontSize: '2rem',
  fontWeight: 'bold',
  color: '#1d3557',
  textAlign: 'center',
  marginBottom: theme.spacing(1),
}));

const StatLabel = styled(Typography)(({ theme }) => ({
  fontSize: '0.9rem',
  color: 'rgba(0, 0, 0, 0.6)',
  textAlign: 'center',
}));

const CharacterCard = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: theme.spacing(2),
  backgroundColor: 'rgba(252, 163, 17, 0.1)',
  borderRadius: theme.shape.borderRadius,
  margin: theme.spacing(0.5),
  width: '5rem',
  height: '7rem',
  justifyContent: 'center',
  transition: 'transform 0.2s, background-color 0.2s',
  '&:hover': {
    transform: 'scale(1.1)',
    backgroundColor: 'rgba(252, 163, 17, 0.2)',
  },
}));

const CharacterDisplay = styled(Typography)({
  fontSize: '2.5rem',
  color: '#1d3557',
  marginBottom: '0.5rem',
});

const FrequencyText = styled(Typography)({
  fontSize: '0.8rem',
  color: 'rgba(0, 0, 0, 0.6)',
});

const BorderLinearProgress = styled(LinearProgress)(({ theme }) => ({
  height: 10,
  borderRadius: 5,
  [`&.MuiLinearProgress-colorPrimary`]: {
    backgroundColor: 'rgba(252, 163, 17, 0.2)',
  },
  [`& .MuiLinearProgress-bar`]: {
    borderRadius: 5,
    backgroundColor: '#fca311',
  },
}));

interface BaybayinStatsProps {
  // Add any props if needed
}

// Updated interface to match the backend response structure
interface BaybayinStatisticsResponse {
  overview: {
    total_words: number;
    with_baybayin: number;
    percentage: number;
  };
  by_language: {
    language_code: string;
    total_words: number;
    with_baybayin: number;
    percentage: number;
  }[];
  character_frequency: {
    [language_code: string]: {
      [character: string]: number;
    };
  };
  completeness: {
    with_baybayin: number;
    without_baybayin: number;
  };
}

// Helper interface for processed character frequency data
interface CharacterFrequencyData {
  character: string;
  frequency: number;
  percentage: number;
}

const BaybayinStatistics: React.FC<BaybayinStatsProps> = () => {
  const [stats, setStats] = useState<BaybayinStatisticsResponse | null>(null);
  const [characterData, setCharacterData] = useState<CharacterFrequencyData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        const data = await getBaybayinStatistics();
        setStats(data);
        
        // Process character frequency data
        const allCharFreq: CharacterFrequencyData[] = [];
        let totalChars = 0;
        
        // Flatten character frequency data from all languages
        Object.entries(data.character_frequency).forEach(([langCode, characters]) => {
          Object.entries(characters).forEach(([char, freq]) => {
            totalChars += freq;
            // Check if character already exists in our array
            const existingIndex = allCharFreq.findIndex(c => c.character === char);
            if (existingIndex >= 0) {
              allCharFreq[existingIndex].frequency += freq;
            } else {
              allCharFreq.push({
                character: char,
                frequency: freq,
                percentage: 0 // Will calculate after all are added
              });
            }
          });
        });
        
        // Calculate percentages and sort by frequency
        allCharFreq.forEach(item => {
          item.percentage = item.frequency / totalChars;
        });
        
        setCharacterData(allCharFreq.sort((a, b) => b.frequency - a.frequency));
      } catch (err) {
        setError('Failed to load Baybayin statistics. Please try again later.');
        console.error('Error fetching Baybayin statistics:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress color="primary" />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!stats) {
    return null;
  }

  // Get total languages with baybayin
  const languagesWithBaybayin = stats.by_language.filter(lang => lang.with_baybayin > 0).length;

  return (
    <StyledPaper>
      <Typography variant="h5" component="h2" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Baybayin Statistics
      </Typography>
      <Divider sx={{ mb: 3 }} />

      {/* Overall Statistics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={4}>
          <StatCard>
            <CardContent>
              <StatValue>{stats.overview.with_baybayin.toLocaleString()}</StatValue>
              <StatLabel>Words with Baybayin</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard>
            <CardContent>
              <StatValue>{stats.overview.percentage.toFixed(1)}%</StatValue>
              <StatLabel>Coverage Percentage</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard>
            <CardContent>
              <StatValue>{languagesWithBaybayin}</StatValue>
              <StatLabel>Languages with Baybayin</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
      </Grid>

      {/* Completeness Comparison */}
      <Typography variant="h6" color="#1d3557" gutterBottom sx={{ fontWeight: 600, mt: 4 }}>
        Completeness Score Comparison
      </Typography>
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6}>
          <StatCard>
            <CardContent>
              <StatValue>{stats.completeness.with_baybayin.toFixed(2)}</StatValue>
              <StatLabel>Average Score with Baybayin</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
        <Grid item xs={12} sm={6}>
          <StatCard>
            <CardContent>
              <StatValue>{stats.completeness.without_baybayin.toFixed(2)}</StatValue>
              <StatLabel>Average Score without Baybayin</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
      </Grid>

      {/* Language Statistics */}
      <Typography variant="h6" color="#1d3557" gutterBottom sx={{ fontWeight: 600, mt: 4 }}>
        Baybayin by Language
      </Typography>
      <TableContainer component={Paper} sx={{ mb: 4 }}>
        <Table>
          <TableHead sx={{ backgroundColor: 'rgba(29, 53, 87, 0.05)' }}>
            <TableRow>
              <TableCell>Language Code</TableCell>
              <TableCell align="right">Total Words</TableCell>
              <TableCell align="right">With Baybayin</TableCell>
              <TableCell>Percentage</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {stats.by_language
              .filter(lang => lang.with_baybayin > 0)
              .sort((a, b) => b.with_baybayin - a.with_baybayin)
              .map((lang) => (
                <TableRow key={lang.language_code}>
                  <TableCell>
                    <Typography variant="body2" fontWeight={500}>
                      {lang.language_code}
                    </Typography>
                  </TableCell>
                  <TableCell align="right">{lang.total_words.toLocaleString()}</TableCell>
                  <TableCell align="right">{lang.with_baybayin.toLocaleString()}</TableCell>
                  <TableCell>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <BorderLinearProgress
                        variant="determinate"
                        value={lang.percentage}
                        sx={{ flexGrow: 1, mr: 2 }}
                      />
                      <Typography variant="body2">{lang.percentage.toFixed(1)}%</Typography>
                    </Box>
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Most Common Characters */}
      <Typography variant="h6" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Most Common Baybayin Characters
      </Typography>
      <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'center', mb: 4 }}>
        {characterData.slice(0, 10).map((char) => (
          <Tooltip 
            key={char.character} 
            title={`${char.frequency} occurrences (${(char.percentage * 100).toFixed(1)}%)`}
          >
            <CharacterCard>
              <CharacterDisplay>{char.character}</CharacterDisplay>
              <FrequencyText>{char.frequency.toLocaleString()}</FrequencyText>
            </CharacterCard>
          </Tooltip>
        ))}
      </Box>
    </StyledPaper>
  );
};

export default BaybayinStatistics; 