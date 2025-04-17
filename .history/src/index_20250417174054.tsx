import React from 'react';
import ReactDOM from 'react-dom';
import './styles/global.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Clear the circuit breaker state on application start to ensure a fresh start
try {
  // Only clear in development mode to avoid clearing in production unnecessarily
  if (process.env.NODE_ENV === 'development') {
    resetCircuitBreaker();
    console.log('Circuit breaker state cleared on startup');
  }
} catch (e) {
  console.error('Error clearing circuit breaker:', e);
}

// Create a query client instance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
      staleTime: 1000 * 60 * 10 // 10 minutes
    }
  }
});

ReactDOM.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </QueryClientProvider>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();