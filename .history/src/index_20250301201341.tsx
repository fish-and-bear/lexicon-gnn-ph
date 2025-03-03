import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/global.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { resetCircuitBreaker } from './api/wordApi';

// Reset circuit breaker on application start
try {
  localStorage.removeItem('circuit_breaker_state');
  console.log('Circuit breaker state cleared on startup');
} catch (e) {
  console.error('Error clearing circuit breaker state:', e);
}

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();