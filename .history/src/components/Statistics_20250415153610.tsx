import React, { useState, useEffect } from 'react';
import { getStatistics, getAdvancedStatistics, getTimeseriesStatistics, getLanguageStatistics } from '../api/wordApi';

// MUI Imports
import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import Divider from '@mui/material/Divider';
import CircularProgress from '@mui/material/CircularProgress';
import Alert from '@mui/material/Alert';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import CardContent from '@mui/material/CardContent';
import { styled } from '@mui/material/styles';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { ResponsiveContainer, LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, PieChart, Pie, Cell } from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
};

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

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658'];

interface StatisticsProps {
  // Add any props if needed
}

const Statistics: React.FC<StatisticsProps> = () => {
  const [tabValue, setTabValue] = useState(0);
  const [basicStats, setBasicStats] = useState<any>(null);
  const [advancedStats, setAdvancedStats] = useState<any>(null);
  const [timeseriesStats, setTimeseriesStats] = useState<any>(null);
  const [languageStats, setLanguageStats] = useState<any>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('fil');
  const [timeInterval, setTimeInterval] = useState<string>('month');
  const [timePeriod, setTimePeriod] = useState<string>('1-year');
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  useEffect(() => {
    const fetchStatistics = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const basicData = await getStatistics();
        setBasicStats(basicData);
      } catch (err) {
        console.error('Error fetching basic statistics:', err);
        setError('Failed to load basic statistics');
      } finally {
        setIsLoading(false);
      }
    };

    fetchStatistics();
  }, []);

  useEffect(() => {
    if (tabValue === 1) {
      const fetchAdvancedStats = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const advancedData = await getAdvancedStatistics();
          setAdvancedStats(advancedData);
        } catch (err) {
          console.error('Error fetching advanced statistics:', err);
          setError('Failed to load advanced statistics');
        } finally {
          setIsLoading(false);
        }
      };

      fetchAdvancedStats();
    }
  }, [tabValue]);

  useEffect(() => {
    if (tabValue === 2) {
      const fetchTimeseriesStats = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const timeseriesData = await getTimeseriesStatistics(timeInterval, timePeriod);
          setTimeseriesStats(timeseriesData);
        } catch (err) {
          console.error('Error fetching timeseries statistics:', err);
          setError('Failed to load timeseries statistics');
        } finally {
          setIsLoading(false);
        }
      };

      fetchTimeseriesStats();
    }
  }, [tabValue, timeInterval, timePeriod]);

  useEffect(() => {
    if (tabValue === 3 && selectedLanguage) {
      const fetchLanguageStats = async () => {
        setIsLoading(true);
        setError(null);
        try {
          const languageData = await getLanguageStatistics(selectedLanguage);
          setLanguageStats(languageData);
        } catch (err) {
          console.error('Error fetching language statistics:', err);
          setError('Failed to load language statistics');
        } finally {
          setIsLoading(false);
        }
      };

      fetchLanguageStats();
    }
  }, [tabValue, selectedLanguage]);

  const renderBasicStats = () => {
    if (!basicStats) return null;

    // Transform data for charts
    const languageData = Object.entries(basicStats.words_by_language || {}).map(([lang, count]) => ({
      name: lang,
      value: count
    }));

    const posData = Object.entries(basicStats.words_by_pos || {}).map(([pos, count]) => ({
      name: pos,
      value: count
    }));

    return (
      <Box>
        <Typography variant="h5" gutterBottom>Basic Dictionary Statistics</Typography>
        
        <Grid container spacing={3} sx={{ mt: 2, mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{basicStats.total_words?.toLocaleString()}</StatValue>
                <StatLabel>Total Words</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{basicStats.total_definitions?.toLocaleString()}</StatValue>
                <StatLabel>Total Definitions</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{basicStats.total_etymologies?.toLocaleString()}</StatValue>
                <StatLabel>Total Etymologies</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{basicStats.total_relations?.toLocaleString()}</StatValue>
                <StatLabel>Total Relations</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
        </Grid>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Words by Language</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={languageData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {languageData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Words by Part of Speech</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={posData.slice(0, 10)}
                  layout="vertical"
                  margin={{ top: 20, right: 30, left: 60, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>Feature Statistics</Typography>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Feature</TableCell>
                  <TableCell align="right">Count</TableCell>
                  <TableCell align="right">Percentage</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell>Words with Baybayin</TableCell>
                  <TableCell align="right">{basicStats.words_with_baybayin?.toLocaleString()}</TableCell>
                  <TableCell align="right">
                    {((basicStats.words_with_baybayin / basicStats.total_words) * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Words with Etymology</TableCell>
                  <TableCell align="right">{basicStats.words_with_etymology?.toLocaleString()}</TableCell>
                  <TableCell align="right">
                    {((basicStats.words_with_etymology / basicStats.total_words) * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Words with Relations</TableCell>
                  <TableCell align="right">{basicStats.words_with_relations?.toLocaleString()}</TableCell>
                  <TableCell align="right">
                    {((basicStats.words_with_relations / basicStats.total_words) * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
                <TableRow>
                  <TableCell>Words with Examples</TableCell>
                  <TableCell align="right">{basicStats.words_with_examples?.toLocaleString()}</TableCell>
                  <TableCell align="right">
                    {((basicStats.words_with_examples / basicStats.total_words) * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      </Box>
    );
  };

  const renderAdvancedStats = () => {
    if (!advancedStats) return null;

    // Transform data for charts
    const completenessData = Object.entries(advancedStats.complexity_distribution || {}).map(([score, count]) => ({
      name: `${score}`,
      value: count
    })).sort((a, b) => parseFloat(a.name) - parseFloat(b.name));

    const definitionCountData = Object.entries(advancedStats.definition_counts || {}).map(([count, wordCount]) => ({
      name: count === '0' ? 'No definitions' : `${count} definition${parseInt(count) !== 1 ? 's' : ''}`,
      value: wordCount
    })).sort((a, b) => {
      const aNum = a.name === 'No definitions' ? 0 : parseInt(a.name.split(' ')[0]);
      const bNum = b.name === 'No definitions' ? 0 : parseInt(b.name.split(' ')[0]);
      return aNum - bNum;
    });

    return (
      <Box>
        <Typography variant="h5" gutterBottom>Advanced Dictionary Statistics</Typography>
        
        <Grid container spacing={3} sx={{ mt: 2, mb: 4 }}>
          <Grid item xs={12} sm={6}>
            <Typography variant="h6" gutterBottom>Language Distribution</Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Language</TableCell>
                    <TableCell align="right">Word Count</TableCell>
                    <TableCell align="right">Percentage</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(advancedStats.language_distribution || {})
                    .sort(([, countA], [, countB]) => (countB as number) - (countA as number))
                    .map(([lang, count]) => (
                      <TableRow key={lang}>
                        <TableCell>{lang}</TableCell>
                        <TableCell align="right">{(count as number).toLocaleString()}</TableCell>
                        <TableCell align="right">
                          {((count as number) / advancedStats.total_words * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
          
          <Grid item xs={12} sm={6}>
            <Typography variant="h6" gutterBottom>Relation Types</Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Relation Type</TableCell>
                    <TableCell align="right">Count</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(advancedStats.relation_types || {})
                    .sort(([, countA], [, countB]) => (countB as number) - (countA as number))
                    .map(([type, count]) => (
                      <TableRow key={type}>
                        <TableCell>{type}</TableCell>
                        <TableCell align="right">{(count as number).toLocaleString()}</TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>
        </Grid>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Completeness Score Distribution</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={completenessData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Bar dataKey="value" fill="#1d3557" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Definition Count Distribution</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={definitionCountData.slice(0, 10)}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Bar dataKey="value" fill="#fca311" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>Baybayin Statistics</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <StatCard>
                <CardContent>
                  <StatValue>{advancedStats.baybayin_stats?.with_baybayin.toLocaleString()}</StatValue>
                  <StatLabel>Words with Baybayin</StatLabel>
                </CardContent>
              </StatCard>
            </Grid>
            <Grid item xs={12} sm={4}>
              <StatCard>
                <CardContent>
                  <StatValue>{advancedStats.baybayin_stats?.percentage.toFixed(1)}%</StatValue>
                  <StatLabel>Baybayin Coverage</StatLabel>
                </CardContent>
              </StatCard>
            </Grid>
            <Grid item xs={12} sm={4}>
              <StatCard>
                <CardContent>
                  <StatValue>{(advancedStats.baybayin_stats?.with_baybayin / (advancedStats.baybayin_stats?.total_words || 1) * 100).toFixed(1)}%</StatValue>
                  <StatLabel>Percentage of Total Words</StatLabel>
                </CardContent>
              </StatCard>
            </Grid>
          </Grid>
        </Box>
      </Box>
    );
  };

  const renderTimeseriesStats = () => {
    if (!timeseriesStats) return null;

    const formatTimeseriesData = (dataKey: string) => {
      if (!timeseriesStats[dataKey]) return [];
      
      return timeseriesStats[dataKey].map((item: any) => ({
        period: item.period_label,
        count: item.word_count || item.definition_count || item.baybayin_count
      }));
    };

    const newWordsData = formatTimeseriesData('new_words');
    const updatedWordsData = formatTimeseriesData('updated_words');
    const newDefinitionsData = formatTimeseriesData('new_definitions');
    const baybayinAdoptionData = formatTimeseriesData('baybayin_adoption');

    return (
      <Box>
        <Typography variant="h5" gutterBottom>Timeseries Statistics</Typography>
        
        <Box sx={{ mb: 3 }}>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Time Interval</InputLabel>
                <Select
                  value={timeInterval}
                  label="Time Interval"
                  onChange={(e) => setTimeInterval(e.target.value)}
                >
                  <MenuItem value="day">Day</MenuItem>
                  <MenuItem value="week">Week</MenuItem>
                  <MenuItem value="month">Month</MenuItem>
                  <MenuItem value="year">Year</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Time Period</InputLabel>
                <Select
                  value={timePeriod}
                  label="Time Period"
                  onChange={(e) => setTimePeriod(e.target.value)}
                >
                  <MenuItem value="1-month">1 Month</MenuItem>
                  <MenuItem value="3-month">3 Months</MenuItem>
                  <MenuItem value="6-month">6 Months</MenuItem>
                  <MenuItem value="1-year">1 Year</MenuItem>
                  <MenuItem value="all-time">All Time</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </Box>

        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>New Words Over Time</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={newWordsData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Line type="monotone" dataKey="count" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Updated Words Over Time</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={updatedWordsData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Line type="monotone" dataKey="count" stroke="#82ca9d" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>New Definitions Over Time</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={newDefinitionsData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Line type="monotone" dataKey="count" stroke="#ffc658" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>Baybayin Adoption Over Time</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={baybayinAdoptionData}
                  margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="period" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Line type="monotone" dataKey="count" stroke="#ff8042" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Grid>
        </Grid>
      </Box>
    );
  };

  const renderLanguageStats = () => {
    if (!languageStats) return null;

    return (
      <Box>
        <Typography variant="h5" gutterBottom>Language-Specific Statistics</Typography>
        
        <Box sx={{ mb: 3 }}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Language</InputLabel>
            <Select
              value={selectedLanguage}
              label="Language"
              onChange={(e) => setSelectedLanguage(e.target.value)}
            >
              <MenuItem value="fil">Filipino</MenuItem>
              <MenuItem value="tgl">Tagalog</MenuItem>
              <MenuItem value="ceb">Cebuano</MenuItem>
              <MenuItem value="hil">Hiligaynon</MenuItem>
              <MenuItem value="ilo">Ilocano</MenuItem>
              <MenuItem value="bik">Bikol</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <Grid container spacing={3} sx={{ mt: 2, mb: 4 }}>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{languageStats.total_words?.toLocaleString()}</StatValue>
                <StatLabel>Total Words</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{languageStats.total_definitions?.toLocaleString()}</StatValue>
                <StatLabel>Total Definitions</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{languageStats.average_completeness?.toFixed(2)}</StatValue>
                <StatLabel>Average Completeness</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <StatCard>
              <CardContent>
                <StatValue>{languageStats.baybayin_percentage?.toFixed(1)}%</StatValue>
                <StatLabel>Baybayin Coverage</StatLabel>
              </CardContent>
            </StatCard>
          </Grid>
        </Grid>

        {languageStats.pos_distribution && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>Parts of Speech Distribution</Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={Object.entries(languageStats.pos_distribution).map(([pos, count]) => ({
                    name: pos,
                    value: count
                  }))}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip formatter={(value) => value.toLocaleString()} />
                  <Bar dataKey="value" fill="#1d3557" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Box>
        )}

        {languageStats.feature_coverage && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" gutterBottom>Feature Coverage</Typography>
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Feature</TableCell>
                    <TableCell align="right">Words with Feature</TableCell>
                    <TableCell align="right">Percentage</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(languageStats.feature_coverage).map(([feature, data]: [string, any]) => (
                    <TableRow key={feature}>
                      <TableCell>{feature}</TableCell>
                      <TableCell align="right">{data.count.toLocaleString()}</TableCell>
                      <TableCell align="right">{data.percentage.toFixed(1)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <StyledPaper>
      <Typography variant="h4" component="h1" color="#1d3557" gutterBottom sx={{ fontWeight: 600 }}>
        Dictionary Statistics
      </Typography>
      <Divider sx={{ mb: 3 }} />

      <Tabs
        value={tabValue}
        onChange={handleTabChange}
        aria-label="statistics tabs"
        sx={{ mb: 2 }}
      >
        <Tab label="Basic Statistics" />
        <Tab label="Advanced Statistics" />
        <Tab label="Timeseries Statistics" />
        <Tab label="Language Statistics" />
      </Tabs>

      {isLoading ? (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress color="primary" />
        </Box>
      ) : error ? (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      ) : (
        <>
          <TabPanel value={tabValue} index={0}>
            {renderBasicStats()}
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            {renderAdvancedStats()}
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            {renderTimeseriesStats()}
          </TabPanel>
          <TabPanel value={tabValue} index={3}>
            {renderLanguageStats()}
          </TabPanel>
        </>
      )}
    </StyledPaper>
  );
};

export default Statistics; 