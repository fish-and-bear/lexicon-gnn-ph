const fs = require('fs');
const data = JSON.parse(fs.readFileSync('data/gay-slang.json', 'utf8'));

// Helper function to split string values based on common Filipino separators
function splitToArray(str) {
  if (!str || typeof str !== 'string') return [];
  
  // First clean the string
  let cleanStr = str.trim();
  if (cleanStr === '') return [];
  
  // Split by common separators: commas, 'o', 'at', semicolons
  // The regex pattern handles multiple formats:
  // "word1, word2" -> ["word1", "word2"]
  // "word1 o word2" -> ["word1", "word2"]
  // "word1 at word2" -> ["word1", "word2"]
  // "word1; word2" -> ["word1", "word2"]
  const parts = cleanStr
    .split(/\s*(?:,|;|\so\s|\sat\s)\s*/)
    .map(part => part.trim())
    .filter(part => part !== '');
  
  return parts.length > 0 ? parts : [cleanStr];
}

data.forEach(item => {
  // Fix null values
  if (item.etymology === null) item.etymology = '';
  if (item.partOfSpeech === null) item.partOfSpeech = '';
  
  // Fix sangkahulugan - ensure it's always an array and split separated values
  if (item.sangkahulugan === null) {
    item.sangkahulugan = [];
  } else if (typeof item.sangkahulugan === 'string') {
    // Split the string by separators and convert to array
    item.sangkahulugan = splitToArray(item.sangkahulugan);
  } else if (Array.isArray(item.sangkahulugan) && item.sangkahulugan.length === 1 && 
             typeof item.sangkahulugan[0] === 'string' && 
             (item.sangkahulugan[0].includes(',') || 
              item.sangkahulugan[0].includes(' o ') || 
              item.sangkahulugan[0].includes(' at '))) {
    // Handle case where separated values are in a single string within an array
    item.sangkahulugan = splitToArray(item.sangkahulugan[0]);
  }
  
  // Fix synonym - ensure it's always an array and split separated values
  if (item.synonym === null) {
    item.synonym = [];
  } else if (typeof item.synonym === 'string') {
    // Split the string by separators and convert to array
    item.synonym = splitToArray(item.synonym);
  } else if (Array.isArray(item.synonym) && item.synonym.length === 1 && 
             typeof item.synonym[0] === 'string' && 
             (item.synonym[0].includes(',') || 
              item.synonym[0].includes(' o ') || 
              item.synonym[0].includes(' at '))) {
    // Handle case where separated values are in a single string within an array
    item.synonym = splitToArray(item.synonym[0]);
  }
  
  // Fix nulls in definitions
  if (item.definitions) {
    item.definitions.forEach(def => {
      if (def.meaning === null) def.meaning = '';
    });
  }
  
  // Ensure variations is always an array
  if (item.variations === null) {
    item.variations = [];
  } else if (typeof item.variations === 'string') {
    item.variations = splitToArray(item.variations);
  } else if (Array.isArray(item.variations) && item.variations.length === 1 && 
             typeof item.variations[0] === 'string' && 
             (item.variations[0].includes(',') || 
              item.variations[0].includes(' o ') || 
              item.variations[0].includes(' at '))) {
    item.variations = splitToArray(item.variations[0]);
  }
  
  // Ensure usageLabels is always an array
  if (item.usageLabels === null || item.usageLabels === undefined) {
    item.usageLabels = [];
  } else if (typeof item.usageLabels === 'string') {
    item.usageLabels = splitToArray(item.usageLabels);
  } else if (Array.isArray(item.usageLabels) && item.usageLabels.length === 1 && 
             typeof item.usageLabels[0] === 'string' && 
             (item.usageLabels[0].includes(',') || 
              item.usageLabels[0].includes(' o ') || 
              item.usageLabels[0].includes(' at '))) {
    item.usageLabels = splitToArray(item.usageLabels[0]);
  }
});

fs.writeFileSync('data/gay-slang.json.fixed', JSON.stringify(data, null, 2));
console.log('Fixed JSON file saved to data/gay-slang.json.fixed'); 