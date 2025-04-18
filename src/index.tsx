import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles/global.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { ThemeProvider } from './contexts/ThemeContext';
import { resetCircuitBreaker } from './api/wordApi';

// --- BEGIN DIAGNOSTIC --- 
// Moved after imports to fix eslint rule
try {
  localStorage.setItem('__test', '1');
  localStorage.removeItem('__test');
  console.log('[DIAG] localStorage access OK.');
} catch (e) {
  console.error('[DIAG] localStorage access FAILED:', e);
  // Optional: Display a message to the user if critical
  // alert('Error accessing localStorage. App may not function correctly.');
}
// --- END DIAGNOSTIC --- 

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

// Use the new createRoot API
const container = document.getElementById('root');
if (container) { // Ensure container is not null
  const root = createRoot(container);
  root.render(
    <React.StrictMode>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </React.StrictMode>
  );
} else {
  console.error("Failed to find the root element. Check public/index.html");
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();