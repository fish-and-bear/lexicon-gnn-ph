/**
 * CSS Backup Utility
 * 
 * This script creates backups of CSS files before modifying them
 */

console.log('Creating CSS file backups...');

const fs = require('fs');
const path = require('path');

// Function to backup a file
function backupFile(filePath) {
  if (fs.existsSync(filePath)) {
    const backupPath = `${filePath}.backup`;
    fs.copyFileSync(filePath, backupPath);
    console.log(`Created backup: ${backupPath}`);
    return true;
  } else {
    console.log(`File not found, skipping backup: ${filePath}`);
    return false;
  }
}

// Backup main CSS files
const filesToBackup = [
  path.join(__dirname, 'index.css'),
  path.join(__dirname, 'App.css'),
  path.join(__dirname, 'styles', 'global.css'),
  path.join(__dirname, 'styles', 'index.css')
];

// Backup all files in the list
filesToBackup.forEach(backupFile);

// Backup all component CSS files
const componentsDir = path.join(__dirname, 'components');
if (fs.existsSync(componentsDir)) {
  const cssFiles = fs.readdirSync(componentsDir).filter(file => file.endsWith('.css'));
  
  cssFiles.forEach(file => {
    const filePath = path.join(componentsDir, file);
    backupFile(filePath);
  });
}

// Backup all styles/components CSS files
const stylesComponentsDir = path.join(__dirname, 'styles', 'components');
if (fs.existsSync(stylesComponentsDir)) {
  const cssFiles = fs.readdirSync(stylesComponentsDir).filter(file => file.endsWith('.css'));
  
  cssFiles.forEach(file => {
    const filePath = path.join(stylesComponentsDir, file);
    backupFile(filePath);
  });
}

console.log('CSS backup complete.'); 