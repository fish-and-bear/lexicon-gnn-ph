/**
 * CSS Restore Utility
 * 
 * This script restores CSS files from backups in case the changes need to be rolled back
 */

console.log('Restoring CSS files from backups...');

const fs = require('fs');
const path = require('path');

// Function to restore a file from backup
function restoreFile(backupPath) {
  if (fs.existsSync(backupPath)) {
    const originalPath = backupPath.replace('.backup', '');
    fs.copyFileSync(backupPath, originalPath);
    console.log(`Restored ${originalPath} from backup`);
    return true;
  } else {
    console.log(`Backup not found, skipping restore: ${backupPath}`);
    return false;
  }
}

// Find all backup files
function findBackupFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  
  list.forEach(file => {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    
    if (stat.isDirectory()) {
      // Recursive search in subdirectories
      results = results.concat(findBackupFiles(filePath));
    } else if (file.endsWith('.backup')) {
      results.push(filePath);
    }
  });
  
  return results;
}

// Find all backups in the src directory
const srcDir = path.join(__dirname);
const backupFiles = findBackupFiles(srcDir);

if (backupFiles.length === 0) {
  console.log('No backup files found.');
} else {
  console.log(`Found ${backupFiles.length} backup files.`);
  
  // Restore all backups
  backupFiles.forEach(restoreFile);
}

console.log('CSS restore complete.'); 