import React, { useState } from 'react';
import { convertToBaybayin } from '../api/wordApi';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Chip from '@mui/material/Chip';
import Divider from '@mui/material/Divider';
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

const BaybayinDisplay = styled(Box)(({ theme }) => ({
  padding: theme.spacing(3),
  marginTop: theme.spacing(3),
  backgroundColor: 'rgba(29, 53, 87, 0.05)',
  borderRadius: theme.shape.borderRadius,
  border: '1px solid rgba(29, 53, 87, 0.1)',
  boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.05)',
  textAlign: 'center',
  minHeight: '4rem',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
}));

const BaybayinText = styled(Typography)(({/* theme */}) => ({
  fontSize: '2.5rem',
  color: 'rgba(29, 53, 87, 0.85)',
  fontFamily: 'Arial, sans-serif',
  letterSpacing: '0.05em',
  animation: 'subtle-pulse 3s ease-in-out infinite',
  '@keyframes subtle-pulse': {
    '0%, 100%': { 
      transform: 'scale(1)', 
      opacity: 0.8 
    },
    '50%': { 
      transform: 'scale(1.05)', 
      opacity: 1 
    },
  },
}));

const ConversionRate = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'flex-end',
  marginTop: theme.spacing(1),
}));

interface BaybayinConverterProps {
  // Add any props if needed
}

const BaybayinConverter: React.FC<BaybayinConverterProps> = () => {
  const [text, setText] = useState('');
  const [languageCode, setLanguageCode] = useState('fil');
  const [result, setResult] = useState<{ 
    original_text: string; 
    baybayin_text: string; 
    conversion_rate: number;
  } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConvert = async () => {
    if (!text.trim()) {
      setError('Please enter some text to convert');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await convertToBaybayin(text, languageCode);
      setResult(data);
    } catch (err) {
      setError('Failed to convert text to Baybayin. Please try again.');
      console.error('Error converting to Baybayin:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <StyledPaper>
      <Typography variant="h5" component="h2" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Baybayin Converter
      </Typography>
      <Divider sx={{ mb: 2 }} />
      
      <Typography variant="body1" gutterBottom sx={{ mb: 2 }}>
        Convert Filipino text to Baybayin script. Words in our dictionary will be converted based on their Baybayin form.
      </Typography>

      <FormControl fullWidth sx={{ mb: 2 }}>
        <InputLabel>Language</InputLabel>
        <Select
          value={languageCode}
          label="Language"
          onChange={(e) => setLanguageCode(e.target.value)}
        >
          <MenuItem value="fil">Filipino</MenuItem>
          <MenuItem value="ceb">Cebuano</MenuItem>
          <MenuItem value="tgl">Tagalog</MenuItem>
        </Select>
      </FormControl>

      <StyledTextField
        fullWidth
        multiline
        rows={3}
        variant="outlined"
        label="Text to convert"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to convert to Baybayin"
      />

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <StyledButton 
        onClick={handleConvert}
        disabled={isLoading || !text.trim()}
        startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : null}
      >
        {isLoading ? 'Converting...' : 'Convert to Baybayin'}
      </StyledButton>

      {result && (
        <>
          <BaybayinDisplay>
            <BaybayinText>{result.baybayin_text}</BaybayinText>
          </BaybayinDisplay>

          <ConversionRate>
            <Chip 
              label={`Conversion rate: ${Math.round(result.conversion_rate * 100)}%`}
              color={result.conversion_rate > 0.7 ? "success" : result.conversion_rate > 0.3 ? "warning" : "error"}
              sx={{ fontWeight: 500 }}
            />
          </ConversionRate>
        </>
      )}
    </StyledPaper>
  );
};

export default BaybayinConverter; 