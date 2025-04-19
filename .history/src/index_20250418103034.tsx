import React from 'react';
import ReactDOM from 'react-dom';
import './styles/index.css'; // Import the main stylesheet directly
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';

// Import performance optimization script
import './performance-fix.js';

// Set a flag to track if we've already cleared circuit breaker
const hasResetCircuitBreaker = localStorage.getItem('circuit_breaker_reset_time');
const currentTime = Date.now();
const ONE_HOUR = 60 * 60 * 1000;

// Clear the circuit breaker state on application start to ensure a fresh start
// Only do this if it hasn't been done in the last hour to avoid performance issues
if (process.env.NODE_ENV === 'development' && (!hasResetCircuitBreaker || (currentTime - parseInt(hasResetCircuitBreaker, 10)) > ONE_HOUR)) {
  try {
    // Defer circuit breaker reset to avoid blocking initial render
    setTimeout(() => {
      resetCircuitBreaker();
      localStorage.setItem('circuit_breaker_reset_time', currentTime.toString());
      console.log('Circuit breaker state cleared on startup');
    }, 3000); // Wait 3 seconds after initial render
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

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();