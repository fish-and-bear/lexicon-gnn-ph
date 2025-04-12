import React from 'react';

interface ErrorDisplayProps {
  message: string;
}

const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ message }) => {
  return (
    <div className="error-details">
      <p>{message}</p>
      <p className="error-suggestion">Try searching for a different word or check your connection.</p>
    </div>
  );
};

export default ErrorDisplay; 