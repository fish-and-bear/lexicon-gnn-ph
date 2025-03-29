const fs = require('fs');
const data = JSON.parse(fs.readFileSync('data/gay-slang.json', 'utf8'));

data.forEach(item => {
  // Fix null values
  if (item.etymology === null) item.etymology = '';
  if (item.partOfSpeech === null) item.partOfSpeech = '';
  
  // Fix sangkahulugan - ensure it's always an array
  if (item.sangkahulugan === null) {
    item.sangkahulugan = [];
  } else if (typeof item.sangkahulugan === 'string') {
    // Convert string to array with the string as a single element
    item.sangkahulugan = item.sangkahulugan.trim() === '' ? [] : [item.sangkahulugan];
  }
  
  // Fix synonym - ensure it's always an array
  if (item.synonym === null) {
    item.synonym = [];
  } else if (typeof item.synonym === 'string') {
    // Convert string to array with the string as a single element
    item.synonym = item.synonym.trim() === '' ? [] : [item.synonym];
  } else if (Array.isArray(item.synonym) && item.synonym.length === 1 && typeof item.synonym[0] === 'string' && item.synonym[0].includes(',')) {
    // Handle case where comma-separated values are in a single string within an array
    item.synonym = item.synonym[0].split(',').map(s => s.trim()).filter(s => s !== '');
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
    item.variations = item.variations.trim() === '' ? [] : [item.variations];
  }
  
  // Ensure usageLabels is always an array
  if (item.usageLabels === null || item.usageLabels === undefined) {
    item.usageLabels = [];
  } else if (typeof item.usageLabels === 'string') {
    item.usageLabels = item.usageLabels.trim() === '' ? [] : [item.usageLabels];
  }
});

fs.writeFileSync('data/gay-slang.json.fixed', JSON.stringify(data, null, 2));
console.log('Fixed JSON file saved to data/gay-slang.json.fixed'); 