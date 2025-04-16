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

interface BaybayinStatistics {
  overall: {
    total_words_with_baybayin: number;
    average_completeness: number;
    total_unique_baybayin_characters: number;
  };
  by_language: {
    language_code: string;
    language_name: string;
    words_with_baybayin: number;
    completeness_score: number;
  }[];
  character_frequency: {
    character: string;
    count: number;
    percentage: number;
  }[];
  completeness_scores: {
    score_range: string;
    count: number;
    percentage: number;
  }[];
}

const BaybayinStatistics: React.FC<BaybayinStatsProps> = () => {
  const [stats, setStats] = useState<BaybayinStatistics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        const data = await getBaybayinStatistics();
        setStats(data);
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
              <StatValue>{stats.overall.total_words_with_baybayin.toLocaleString()}</StatValue>
              <StatLabel>Words with Baybayin</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard>
            <CardContent>
              <StatValue>{(stats.overall.average_completeness * 100).toFixed(1)}%</StatValue>
              <StatLabel>Average Completeness</StatLabel>
            </CardContent>
          </StatCard>
        </Grid>
        <Grid item xs={12} sm={4}>
          <StatCard>
            <CardContent>
              <StatValue>{stats.overall.total_unique_baybayin_characters}</StatValue>
              <StatLabel>Unique Characters</StatLabel>
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
              <TableCell>Language</TableCell>
              <TableCell align="right">Words with Baybayin</TableCell>
              <TableCell align="right">Completeness</TableCell>
              <TableCell>Progress</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {stats.by_language.map((lang) => (
              <TableRow key={lang.language_code}>
                <TableCell>
                  <Typography variant="body2" fontWeight={500}>
                    {lang.language_name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {lang.language_code}
                  </Typography>
                </TableCell>
                <TableCell align="right">{lang.words_with_baybayin.toLocaleString()}</TableCell>
                <TableCell align="right">{(lang.completeness_score * 100).toFixed(1)}%</TableCell>
                <TableCell>
                  <BorderLinearProgress
                    variant="determinate"
                    value={lang.completeness_score * 100}
                  />
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
        {stats.character_frequency.slice(0, 10).map((char) => (
          <Tooltip 
            key={char.character} 
            title={`${char.count} occurrences (${(char.percentage * 100).toFixed(1)}%)`}
          >
            <CharacterCard>
              <CharacterDisplay>{char.character}</CharacterDisplay>
              <FrequencyText>{(char.percentage * 100).toFixed(1)}%</FrequencyText>
            </CharacterCard>
          </Tooltip>
        ))}
      </Box>

      {/* Completeness Distribution */}
      <Typography variant="h6" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Completeness Distribution
      </Typography>
      <TableContainer component={Paper}>
        <Table>
          <TableHead sx={{ backgroundColor: 'rgba(29, 53, 87, 0.05)' }}>
            <TableRow>
              <TableCell>Score Range</TableCell>
              <TableCell align="right">Word Count</TableCell>
              <TableCell align="right">Percentage</TableCell>
              <TableCell>Distribution</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {stats.completeness_scores.map((score) => (
              <TableRow key={score.score_range}>
                <TableCell>{score.score_range}</TableCell>
                <TableCell align="right">{score.count.toLocaleString()}</TableCell>
                <TableCell align="right">{(score.percentage * 100).toFixed(1)}%</TableCell>
                <TableCell>
                  <BorderLinearProgress
                    variant="determinate"
                    value={score.percentage * 100}
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </StyledPaper>
  );
};

export default BaybayinStatistics; 