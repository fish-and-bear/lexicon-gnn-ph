import React, { useState, useEffect } from 'react';
import { getQualityAssessment } from '../api/wordApi';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Grid from '@mui/material/Grid';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import CardHeader from '@mui/material/CardHeader';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import Chip from '@mui/material/Chip';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  backgroundColor: '#f8f9fa',
  border: '1px solid rgba(0, 0, 0, 0.1)',
}));

const StyledCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  border: '1px solid rgba(0, 0, 0, 0.1)',
}));

const StyledButton = styled(Button)(({ theme }) => ({
  backgroundColor: '#1d3557',
  color: 'white',
  '&:hover': {
    backgroundColor: '#152538',
  },
  marginTop: theme.spacing(2),
}));

const CompletionBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginTop: theme.spacing(1),
}));

const CompletionLabel = styled(Typography)(({ theme }) => ({
  minWidth: '100px',
  color: 'rgba(0, 0, 0, 0.6)',
}));

const CompletenessBar = styled(Box)(({ 
  theme, 
  completeness = 0,
}: {
  theme?: any,
  completeness: number
}) => ({
  flex: 1,
  backgroundColor: '#e0e0e0',
  borderRadius: theme.shape.borderRadius,
  height: '12px',
  overflow: 'hidden',
  '&::after': {
    content: '""',
    display: 'block',
    height: '100%',
    width: `${completeness * 100}%`,
    backgroundColor: completeness < 0.3 ? '#f44336' : 
                    completeness < 0.6 ? '#ff9800' : 
                    completeness < 0.8 ? '#ffc107' : '#4caf50',
    transition: 'width 0.5s ease-in-out',
  }
}));

const CompletionPercentage = styled(Typography)(({ theme }) => ({
  marginLeft: theme.spacing(1),
  minWidth: '50px',
  textAlign: 'right',
}));

interface QualityAssessmentProps {
  // Add any props if needed
}

interface QualityFilters {
  language_code: string;
  pos: string;
  min_completeness: number;
  max_completeness: number;
  include_issues: boolean;
  issue_severity: string;
  max_results: number;
}

const QualityAssessment: React.FC<QualityAssessmentProps> = () => {
  const [filters, setFilters] = useState<QualityFilters>({
    language_code: '',
    pos: '',
    min_completeness: 0,
    max_completeness: 1,
    include_issues: true,
    issue_severity: 'all',
    max_results: 100
  });
  
  const [results, setResults] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [completenessRange, setCompletenessRange] = useState<number[]>([0, 100]);

  const handleCompletenessChange = (_: Event, newValue: number | number[]) => {
    if (Array.isArray(newValue)) {
      setCompletenessRange(newValue);
      setFilters({
        ...filters,
        min_completeness: newValue[0] / 100,
        max_completeness: newValue[1] / 100
      });
    }
  };

  const handleFilterChange = (field: keyof QualityFilters, value: any) => {
    setFilters({ ...filters, [field]: value });
  };

  const fetchQualityAssessment = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const data = await getQualityAssessment(filters);
      setResults(data);
    } catch (err) {
      console.error('Error fetching quality assessment:', err);
      setError('Failed to load quality assessment data');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchQualityAssessment();
  }, []);

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const renderIssuesList = (word: any) => {
    if (!word.issues || word.issues.length === 0) {
      return (
        <Alert severity="success" sx={{ mt: 1 }}>
          No issues found for this word.
        </Alert>
      );
    }

    return (
      <List dense>
        {word.issues.map((issue: any, index: number) => (
          <ListItem key={index}>
            <ListItemIcon>
              {getSeverityIcon(issue.severity)}
            </ListItemIcon>
            <ListItemText
              primary={issue.issue_type}
              secondary={issue.description}
            />
            <Chip 
              label={issue.severity} 
              size="small" 
              color={getSeverityColor(issue.severity) as any}
              variant="outlined"
            />
          </ListItem>
        ))}
      </List>
    );
  };

  const renderWordCard = (word: any) => {
    return (
      <StyledCard key={word.id}>
        <CardHeader
          title={word.lemma}
          subheader={`${word.language_code} • ${word.pos || 'Unknown POS'}`}
          action={
            <Chip 
              label={`${(word.completeness_score * 100).toFixed(0)}% Complete`}
              color={
                word.completeness_score < 0.3 ? 'error' : 
                word.completeness_score < 0.6 ? 'warning' : 
                word.completeness_score < 0.8 ? 'info' : 'success'
              }
            />
          }
        />
        <CardContent>
          {word.definition_summary && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2">Definition:</Typography>
              <Typography variant="body2" color="text.secondary">
                {word.definition_summary}
              </Typography>
            </Box>
          )}
          
          <Typography variant="subtitle2">Completeness Breakdown:</Typography>
          <Grid container spacing={1} sx={{ mt: 0.5 }}>
            {word.feature_scores && Object.entries(word.feature_scores).map(([feature, score]: [string, any]) => (
              <Grid item xs={12} sm={6} key={feature}>
                <CompletionBox>
                  <CompletionLabel variant="caption">{feature}</CompletionLabel>
                  <CompletenessBar completeness={score} />
                  <CompletionPercentage variant="caption">{(score * 100).toFixed(0)}%</CompletionPercentage>
                </CompletionBox>
              </Grid>
            ))}
          </Grid>
          
          <Typography variant="subtitle2" sx={{ mt: 2 }}>Issues:</Typography>
          {renderIssuesList(word)}
        </CardContent>
      </StyledCard>
    );
  };

  return (
    <StyledPaper>
      <Typography variant="h4" component="h1" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Quality Assessment
      </Typography>
      <Divider sx={{ mb: 3 }} />
      
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth>
            <InputLabel>Language</InputLabel>
            <Select
              value={filters.language_code}
              label="Language"
              onChange={(e) => handleFilterChange('language_code', e.target.value)}
            >
              <MenuItem value="">All Languages</MenuItem>
              <MenuItem value="fil">Filipino</MenuItem>
              <MenuItem value="tgl">Tagalog</MenuItem>
              <MenuItem value="ceb">Cebuano</MenuItem>
              <MenuItem value="hil">Hiligaynon</MenuItem>
              <MenuItem value="ilo">Ilocano</MenuItem>
              <MenuItem value="bik">Bikol</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth>
            <InputLabel>Part of Speech</InputLabel>
            <Select
              value={filters.pos}
              label="Part of Speech"
              onChange={(e) => handleFilterChange('pos', e.target.value)}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="n">Noun</MenuItem>
              <MenuItem value="v">Verb</MenuItem>
              <MenuItem value="adj">Adjective</MenuItem>
              <MenuItem value="adv">Adverb</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth>
            <InputLabel>Issue Severity</InputLabel>
            <Select
              value={filters.issue_severity}
              label="Issue Severity"
              onChange={(e) => handleFilterChange('issue_severity', e.target.value)}
            >
              <MenuItem value="all">All Issues</MenuItem>
              <MenuItem value="critical">Critical Only</MenuItem>
              <MenuItem value="warning">Warning & Critical</MenuItem>
              <MenuItem value="info">Info & Above</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <FormControl fullWidth>
            <InputLabel>Max Results</InputLabel>
            <Select
              value={filters.max_results}
              label="Max Results"
              onChange={(e) => handleFilterChange('max_results', e.target.value)}
            >
              <MenuItem value={10}>10</MenuItem>
              <MenuItem value={25}>25</MenuItem>
              <MenuItem value={50}>50</MenuItem>
              <MenuItem value={100}>100</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          <Typography variant="subtitle2" gutterBottom>
            Completeness Score Range: {completenessRange[0]}% - {completenessRange[1]}%
          </Typography>
          <Slider
            value={completenessRange}
            onChange={handleCompletenessChange}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => `${value}%`}
            sx={{ width: '100%' }}
          />
        </Grid>
      </Grid>
      
      <StyledButton onClick={fetchQualityAssessment} variant="contained">
        Apply Filters
      </StyledButton>
      
      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress color="primary" />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mt: 2 }}>
          {error}
        </Alert>
      ) : results ? (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Results ({results.total_assessed} words assessed)
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h4" align="center" color="#1d3557">
                    {results.average_completeness ? (results.average_completeness * 100).toFixed(1) : 0}%
                  </Typography>
                  <Typography variant="subtitle2" align="center">Average Completeness</Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h4" align="center" color="#f44336">
                    {results.issues_summary?.critical || 0}
                  </Typography>
                  <Typography variant="subtitle2" align="center">Critical Issues</Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h4" align="center" color="#ff9800">
                    {results.issues_summary?.warning || 0}
                  </Typography>
                  <Typography variant="subtitle2" align="center">Warning Issues</Typography>
                </CardContent>
              </Card>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Card>
                <CardContent>
                  <Typography variant="h4" align="center" color="#2196f3">
                    {results.issues_summary?.info || 0}
                  </Typography>
                  <Typography variant="subtitle2" align="center">Info Issues</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
          
          {results.words && results.words.length > 0 ? (
            <Box>
              {results.words.map((word: any) => renderWordCard(word))}
            </Box>
          ) : (
            <Alert severity="info" sx={{ mt: 2 }}>
              No words found matching the current filters.
            </Alert>
          )}
        </Box>
      ) : null}
    </StyledPaper>
  );
};

export default QualityAssessment; 