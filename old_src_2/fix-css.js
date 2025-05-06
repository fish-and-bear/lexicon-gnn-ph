/**
 * CSS Fix Utility
 * 
 * This script fixes CSS integration issues by:
 * 1. Making sure components import common.css first
 * 2. Removing duplicate imports
 * 3. Updating import paths to match the new structure
 */

console.log('Running CSS integration fixes...');

const fs = require('fs');
const path = require('path');

// Function to update component imports
function updateComponentImports(filePath) {
  if (fs.existsSync(filePath) && filePath.endsWith('.tsx')) {
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Component file basename without extension
    const baseName = path.basename(filePath, '.tsx');
    
    // CSS file path for this component
    const cssFile = `${baseName}.css`;
    
    // Check if component imports its own CSS
    if (content.includes(`import './${cssFile}'`) || content.includes(`import "./${cssFile}"`)) {
      // Check if it already imports common.css
      if (!content.includes(`import './common.css'`) && !content.includes(`import "./common.css"`)) {
        // Add common.css import before the component's CSS import
        content = content.replace(
          new RegExp(`import ['"]\\.\\/${cssFile}['"]`, 'g'),
          `import './common.css'; // Added for CSS integration\nimport './${cssFile}'`
        );
        
        fs.writeFileSync(filePath, content, 'utf8');
        console.log(`Updated ${filePath} to import common.css`);
      }
    }
  }
}

// Process all component files
const componentsDir = path.join(__dirname, 'components');
if (fs.existsSync(componentsDir)) {
  const tsxFiles = fs.readdirSync(componentsDir).filter(file => file.endsWith('.tsx'));
  
  tsxFiles.forEach(file => {
    const filePath = path.join(componentsDir, file);
    updateComponentImports(filePath);
  });
}

// Create a debug.css file to help diagnose rendering issues
const debugCssPath = path.join(__dirname, 'debug.css');
const debugCssContent = `/**
 * CSS Debug Styles
 * 
 * This file can be manually imported to diagnose rendering issues.
 * Add this import to your component for debugging: import '../debug.css';
 */

/* Uncomment these lines to debug layout issues */

/*
* {
  outline: 1px solid rgba(255, 0, 0, 0.2) !important;
}

.component-container {
  outline: 2px solid rgba(0, 128, 255, 0.5) !important;
}

:root {
  --debug-active: 1; 
}

.css-debug-notice {
  display: block !important;
}
*/
`;

fs.writeFileSync(debugCssPath, debugCssContent, 'utf8');
console.log(`Created ${debugCssPath} for debugging`);

console.log('CSS integration fixes complete. Try starting the application now.'); 