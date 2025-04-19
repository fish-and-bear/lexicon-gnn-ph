import React from 'react';
import ReactDOM from 'react-dom';
import './styles/index.css'; // Keep basic styles

console.log('src/index.tsx: Core imports loaded, before any rendering.');

// Everything else commented out
/*
import App from './App';
// import { ThemeProvider } from './contexts/ThemeContext'; // Keep ThemeProvider commented out
// import reportWebVitals from './reportWebVitals'; // Keep commented out
// import { resetCircuitBreaker } from './api/wordApi'; // Keep commented out

// Keep circuit breaker logic commented out


try {
  ReactDOM.render(
    <React.StrictMode>
      {/* <ThemeProvider> */} {/* Render App without ThemeProvider */}
        <App />
      {/* </ThemeProvider> */}
    </React.StrictMode>,
    document.getElementById('root')
  );
  console.log('index.tsx: Direct App render call succeeded.');
} catch (error) {
  console.error('index.tsx: Direct App render call FAILED:', error);
}

// Keep reportWebVitals commented out

*/