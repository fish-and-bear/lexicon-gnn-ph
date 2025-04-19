import React from 'react';
import ReactDOM from 'react-dom';
import './styles/index.css'; // Keep basic styles
import reportWebVitals from './reportWebVitals';
import { resetCircuitBreaker } from './api/wordApi';

console.log('src/index.tsx: Core imports + reportWebVitals + wordApi import loaded.');

// Everything else is removed or commented out for this test
/*
import App from './App';
import { ThemeProvider } from './contexts/ThemeContext';

const hasResetCircuitBreaker = localStorage.getItem('circuit_breaker_reset_time');
const currentTime = Date.now();
const ONE_HOUR = 60 * 60 * 1000;

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
  reportWebVitals(console.log);
} catch (error) {
  console.error('index.tsx: Error calling reportWebVitals:', error);
}
*/