import React from 'react';
// REMOVED: console.log('index.tsx loaded');
import ReactDOM from 'react-dom';
import './styles/index.css'; // Keep basic styles

console.log('Attempting basic React render...');

try {
  ReactDOM.render(
    <React.StrictMode>
      <h1>Test Render</h1>
    </React.StrictMode>,
    document.getElementById('root')
  );
  console.log('Basic React render succeeded.');
} catch (error) {
  console.error('Basic React render FAILED:', error);
}

// Commented out original app logic
/*
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';

// Set a flag to track if we've already cleared circuit breaker
const hasResetCircuitBreaker = localStorage.getItem('circuit_breaker_reset_time');
const currentTime = Date.now();
const ONE_HOUR = 60 * 60 * 1000;

// Clear the circuit breaker state on application start
if (process.env.NODE_ENV === 'development' && (!hasResetCircuitBreaker || (currentTime - parseInt(hasResetCircuitBreaker, 10)) > ONE_HOUR)) {
  try {
    setTimeout(() => {
      resetCircuitBreaker();
      localStorage.setItem('circuit_breaker_reset_time', currentTime.toString());
      console.log('Circuit breaker state cleared on startup');
    }, 3000);
  } catch (e) {
    console.error('Error clearing circuit breaker:', e);
  }
}

ReactDOM.render(
  <React.StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </React.StrictMode>,
  document.getElementById('root')
);

reportWebVitals();
*/