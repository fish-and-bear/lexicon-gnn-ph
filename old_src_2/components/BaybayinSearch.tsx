import React, { useState, useEffect } from 'react';
import { searchBaybayin } from '../api/wordApi';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Divider from '@mui/material/Divider';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import Pagination from '@mui/material/Pagination';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  marginBottom: theme.spacing(3),
  borderRadius: theme.shape.borderRadius,
  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
  backgroundColor: '#f8f9fa',
  border: '1px solid rgba(0, 0, 0, 0.1)',
}));

const StyledTextField = styled(TextField)(({ theme }) => ({
  marginBottom: theme.spacing(2),
  '& .MuiOutlinedInput-root': {
    backgroundColor: '#ffffff',
    '&:hover fieldset': {
      borderColor: '#fca311',
    },
    '&.Mui-focused fieldset': {
      borderColor: '#1d3557',
    },
  },
}));

const StyledButton = styled(Button)(({ theme }) => ({
  backgroundColor: '#fca311',
  color: 'white',
  '&:hover': {
    backgroundColor: '#e69b00',
  },
  padding: theme.spacing(1, 3),
  boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
}));

const BaybayinChar = styled(Typography)(({ theme }) => ({
  fontSize: '1.75rem',
  fontFamily: 'Arial, sans-serif',
  color: '#1d3557',
}));

interface BaybayinSearchProps {
  // Add any props if needed
}

interface SearchResult {
  id: number;
  lemma: string;
  language_code: string;
  baybayin_form: string;
  pos: string;
  completeness_score: number;
}

interface SearchResponse {
  count: number;
  results: SearchResult[];
}

const BaybayinSearch: React.FC<BaybayinSearchProps> = () => {
  const [query, setQuery] = useState('');
  const [languageCode, setLanguageCode] = useState('');
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  
  const resultsPerPage = 10;

  const handleSearch = async () => {
    if (!query.trim()) {
      setValidationError('Please enter a search query');
      return;
    }

    // Check if query contains at least one Baybayin character
    if (!/[\u1700-\u171F]/.test(query)) {
      setValidationError('Query must contain at least one Baybayin character');
      return;
    }

    setValidationError(null);
    setIsLoading(true);
    setError(null);

    try {
      const data = await searchBaybayin(query, {
        language_code: languageCode || undefined,
        limit: resultsPerPage,
        offset: (page - 1) * resultsPerPage
      });
      setResults(data);
    } catch (err) {
      setError('Failed to search for Baybayin words. Please try again.');
      console.error('Error searching Baybayin:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Search when page changes
  useEffect(() => {
    if (results && query) {
      handleSearch();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [page]);

  const handlePageChange = (_event: React.ChangeEvent<unknown>, value: number) => {
    setPage(value);
  };

  return (
    <StyledPaper>
      <Typography variant="h5" component="h2" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Baybayin Search
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <Typography variant="body1" gutterBottom sx={{ mb: 2 }}>
        Search for words containing specific Baybayin characters. Enter at least one Baybayin character in your query.
      </Typography>

      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <StyledTextField
          fullWidth
          variant="outlined"
          label="Baybayin Search Query"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter Baybayin characters to search"
        />
        
        <FormControl sx={{ minWidth: 150 }}>
          <InputLabel>Language</InputLabel>
          <Select
            value={languageCode}
            label="Language"
            onChange={(e) => setLanguageCode(e.target.value)}
          >
            <MenuItem value="">All Languages</MenuItem>
            <MenuItem value="fil">Filipino</MenuItem>
            <MenuItem value="tgl">Tagalog</MenuItem>
            <MenuItem value="ceb">Cebuano</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {validationError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {validationError}
        </Alert>
      )}

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <StyledButton 
        onClick={handleSearch}
        disabled={isLoading || !query.trim()}
        startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : null}
      >
        {isLoading ? 'Searching...' : 'Search'}
      </StyledButton>

      {results && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" color="#1d3557" gutterBottom>
            Search Results {results.count > 0 ? `(${results.count} words found)` : ''}
          </Typography>
          
          {results.count > 0 ? (
            <>
              <TableContainer component={Paper} sx={{ mb: 2 }}>
                <Table>
                  <TableHead sx={{ backgroundColor: 'rgba(29, 53, 87, 0.05)' }}>
                    <TableRow>
                      <TableCell>Word</TableCell>
                      <TableCell>Baybayin Form</TableCell>
                      <TableCell>Language</TableCell>
                      <TableCell align="right">Completeness</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {results.results.map((result) => (
                      <TableRow key={result.id}>
                        <TableCell>
                          <Typography variant="body1" fontWeight={500}>
                            {result.lemma}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {result.pos}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <BaybayinChar>{result.baybayin_form}</BaybayinChar>
                        </TableCell>
                        <TableCell>{result.language_code}</TableCell>
                        <TableCell align="right">
                          {(result.completeness_score * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
              
              {results.count > resultsPerPage && (
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                  <Pagination 
                    count={Math.ceil(results.count / resultsPerPage)} 
                    page={page}
                    onChange={handlePageChange}
                    color="primary"
                  />
                </Box>
              )}
            </>
          ) : (
            <Alert severity="info">
              No words found with the given Baybayin characters.
            </Alert>
          )}
        </Box>
      )}
    </StyledPaper>
  );
};

export default BaybayinSearch; 