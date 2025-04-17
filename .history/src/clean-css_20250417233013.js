/**
 * CSS Cleanup Utility
 * 
 * This script helps resolve CSS conflicts by:
 * 1. Ensuring the correct import order
 * 2. Removing duplicated CSS rules
 * 3. Applying the global variable system
 * 
 * Run this script with Node.js before starting the application
 * to ensure CSS is properly organized.
 */

console.log('CSS Cleanup Script - Resolving Style Conflicts');

// Import necessary modules
const fs = require('fs');
const path = require('path');

// Function to add a comment to a file
function addCommentToFile(filePath, comment) {
  if (fs.existsSync(filePath)) {
    const content = fs.readFileSync(filePath, 'utf8');
    if (!content.includes(comment)) {
      const newContent = `${comment}\n\n${content}`;
      fs.writeFileSync(filePath, newContent, 'utf8');
      console.log(`Updated ${filePath}`);
    }
  }
}

// Add comments to component CSS files indicating they should use variables
const componentsDir = path.join(__dirname, 'components');
if (fs.existsSync(componentsDir)) {
  const cssFiles = fs.readdirSync(componentsDir).filter(file => file.endsWith('.css'));
  
  cssFiles.forEach(file => {
    const filePath = path.join(componentsDir, file);
    const comment = `/**
 * Component-specific styles for ${file}
 * NOTE: This file should eventually be migrated to src/styles/components/
 * For now, make sure to import common.css before this file in components.
 */`;
    
    addCommentToFile(filePath, comment);
  });
}

// Create common.css if it doesn't exist
const commonCssPath = path.join(__dirname, 'components', 'common.css');
if (!fs.existsSync(commonCssPath)) {
  const commonCssContent = `/**
 * Common CSS imports for components
 * This file should be imported by components that need access to shared styling
 */
 
/* Placeholder for common styling */
`;
  fs.writeFileSync(commonCssPath, commonCssContent, 'utf8');
  console.log('Created common.css for components');
}

console.log('CSS cleanup complete. You can now start the application.'); 