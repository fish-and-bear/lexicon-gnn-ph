import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';

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