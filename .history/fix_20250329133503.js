const fs = require('fs');
const data = JSON.parse(fs.readFileSync('data/gay-slang.json', 'utf8'));

data.forEach(item => {
  // Fix null values
  if (item.etymology === null) item.etymology = '';
  if (item.sangkahulugan === null) item.sangkahulugan = [];
  if (item.synonym === null) item.synonym = [];
  
  // Fix nulls in definitions
  if (item.definitions) {
    item.definitions.forEach(def => {
      if (def.meaning === null) def.meaning = '';
    });
  }
});

fs.writeFileSync('data/gay-slang.json.fixed', JSON.stringify(data, null, 2));
console.log('Fixed JSON file saved to data/gay-slang.json.fixed'); 