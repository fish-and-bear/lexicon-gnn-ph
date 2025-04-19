import React from 'react';
// REMOVED: console.log('index.tsx loaded');
import ReactDOM from 'react-dom';
import './styles/index.css'; // Keep basic styles
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';

// REMOVED: import './performance-fix.js'; // Keep this removed

console.log('index.tsx: All imports loaded'); // Keep a log

// COMMENTING OUT Circuit Breaker Logic
/*
// Set a flag to track if we've already cleared circuit breaker
const hasResetCircuitBreaker = localStorage.getItem('circuit_breaker_reset_time');
const currentTime = Date.now();
const ONE_HOUR = 60 * 60 * 1000;

// Clear the circuit breaker state on application start
if (process.env.NODE_ENV === 'development' && (!hasResetCircuitBreaker || (currentTime - parseInt(hasResetCircuitBreaker || '0', 10)) > ONE_HOUR)) {
  console.log('index.tsx: Attempting circuit breaker reset...');
  try {
    setTimeout(() => {
      try {
        resetCircuitBreaker();
        localStorage.setItem('circuit_breaker_reset_time', currentTime.toString());
        console.log('index.tsx: Circuit breaker state cleared successfully.');
      } catch (e) {
        console.error('index.tsx: Error during async circuit breaker reset:', e);
      }
    }, 3000);
  } catch (e) {
    console.error('index.tsx: Error setting up circuit breaker reset timeout:', e);
  }
}
*/

console.log('index.tsx: Attempting to render App component...');

try {
  ReactDOM.render(
    <React.StrictMode>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </React.StrictMode>,
    document.getElementById('root')
  );
  console.log('index.tsx: ReactDOM.render called successfully.');
} catch (error) {
  console.error('index.tsx: Error during ReactDOM.render:', error);
}

console.log('index.tsx: Calling reportWebVitals...');
try {
  reportWebVitals(console.log); // Log web vitals to console
} catch (error) {
  console.error('index.tsx: Error calling reportWebVitals:', error);
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