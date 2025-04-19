import React from 'react';
import axios from 'axios'; // Keep axios import
// Import from the dummy wordApi.ts - no specific functions needed for this minimal test
import * as wordApi from "../api/wordApi"; 

// All other imports, state, effects, functions, logic, and JSX are removed.

const WordExplorer: React.FC = () => {
  // Log access to the imported module to ensure it's loaded, even if unused.
  console.log('>>> Minimal WordExplorer component rendering. wordApi module:', wordApi);
  console.log('>>> Axios module:', axios);

  return (
    <div>
      Minimal WordExplorer Test - Check Console for logs.
    </div>
  );
};

export default WordExplorer;
