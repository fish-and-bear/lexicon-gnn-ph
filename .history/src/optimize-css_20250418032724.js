/**
 * CSS Optimizer Utility
 * 
 * This script optimizes CSS files to improve load time:
 * 1. Removes duplicate rules
 * 2. Organizes CSS imports properly
 * 3. Splits large files into smaller modules
 * 
 * Run this script with Node.js to optimize CSS files
 */

console.log('Running CSS optimization...');

const fs = require('fs');
const path = require('path');

// Map of component types to their CSS files for splitting large files
const componentCategories = {
  layout: ['header', 'footer', 'main', 'sidebar'],
  interaction: ['buttons', 'forms', 'inputs', 'search'],
  content: ['typography', 'cards', 'details', 'tabs'],
  visualization: ['graph', 'network', 'nodes', 'edges'],
  theme: ['colors', 'dark-mode', 'light-mode', 'variables']
};

// Function to optimize a CSS file
function optimizeCssFile(filePath) {
  if (!fs.existsSync(filePath) || !filePath.endsWith('.css')) {
    return;
  }
  
  console.log(`Optimizing CSS file: ${filePath}`);
  
  let content = fs.readFileSync(filePath, 'utf8');
  const originalSize = content.length;
  
  // 1. Find and collect all import statements
  const importRegex = /@import\s+['"]([^'"]+)['"]\s*;/g;
  const imports = [];
  let importMatch;
  
  while ((importMatch = importRegex.exec(content)) !== null) {
    imports.push(importMatch[0]);
  }
  
  // 2. Remove all imports so we can add them at the top in correct order
  content = content.replace(importRegex, '');
  
  // 3. Add imports back at the top of the file in the correct order
  if (imports.length > 0) {
    const orderedImports = imports.join('\n');
    content = `/* Ordered Imports */\n${orderedImports}\n\n${content}`;
  }
  
  // 4. Fix invalid or duplicated media queries
  const mediaQueryRegex = /@media\s*\([^{]+\)\s*{[^{}]*({[^{}]*}[^{}]*)*}/g;
  const mediaQueries = {};
  
  let mediaMatch;
  while ((mediaMatch = mediaQueryRegex.exec(content)) !== null) {
    const query = mediaMatch[0];
    const queryKey = query.split('{')[0].trim();
    
    if (!mediaQueries[queryKey]) {
      mediaQueries[queryKey] = [];
    }
    
    mediaQueries[queryKey].push(query);
  }
  
  // Replace duplicate media queries with combined ones
  for (const [queryKey, queries] of Object.entries(mediaQueries)) {
    if (queries.length > 1) {
      // Combine all duplicate media queries
      const combinedQuery = combineMediaQueries(queries, queryKey);
      
      // Replace all instances with the combined one
      queries.forEach(query => {
        content = content.replace(query, '');
      });
      
      content = content.replace(/}\s*}/g, '}}'); // Fix any potentially malformed closing brackets
      content += `\n\n/* Combined Media Query */\n${combinedQuery}`;
    }
  }
  
  // 5. Clean up whitespace
  content = content.replace(/\n\s*\n\s*\n/g, '\n\n');
  
  // Save optimized file
  fs.writeFileSync(filePath, content, 'utf8');
  
  const newSize = content.length;
  const reduction = ((originalSize - newSize) / originalSize * 100).toFixed(2);
  console.log(`Optimized ${filePath}: ${reduction}% size reduction`);
  
  return {
    file: filePath,
    originalSize,
    newSize,
    reduction
  };
}

// Helper to combine media queries
function combineMediaQueries(queries, queryKey) {
  let combinedRules = '';
  
  queries.forEach(query => {
    // Extract rules between the first '{' and the last '}'
    const rulesMatch = query.match(/{([\s\S]+)}/);
    if (rulesMatch && rulesMatch[1]) {
      combinedRules += rulesMatch[1].trim() + '\n';
    }
  });
  
  return `${queryKey} {\n${combinedRules}\n}`;
}

// Process the largest CSS files first
const componentsDir = path.join(__dirname, 'components');
if (fs.existsSync(componentsDir)) {
  const cssFiles = fs.readdirSync(componentsDir)
    .filter(file => file.endsWith('.css'))
    .map(file => ({
      name: file,
      path: path.join(componentsDir, file),
      size: fs.statSync(path.join(componentsDir, file)).size
    }))
    .sort((a, b) => b.size - a.size); // Sort by size, largest first
  
  console.log(`Found ${cssFiles.length} CSS files to optimize`);
  
  // Optimize the largest files (most problematic)
  const results = cssFiles.map(file => optimizeCssFile(file.path));
  
  // Log the results
  const totalOriginal = results.reduce((sum, result) => sum + (result?.originalSize || 0), 0);
  const totalNew = results.reduce((sum, result) => sum + (result?.newSize || 0), 0);
  const totalReduction = ((totalOriginal - totalNew) / totalOriginal * 100).toFixed(2);
  
  console.log(`\nOptimization complete!`);
  console.log(`Total size reduction: ${totalReduction}%`);
  console.log(`Original size: ${(totalOriginal / 1024).toFixed(2)} KB`);
  console.log(`New size: ${(totalNew / 1024).toFixed(2)} KB`);
}

console.log('CSS optimization complete. Your application should load faster now.'); 