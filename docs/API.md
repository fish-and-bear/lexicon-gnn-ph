# FilRelex API Documentation

## Overview

The FilRelex API provides comprehensive access to a Filipino dictionary database with robust support for Baybayin script, word relationships, etymologies, and linguistic data analysis. This RESTful API delivers powerful search capabilities, statistical insights, and specialized Baybayin functionality.

## Base URL

```
https://api.fil-relex.example.com/api/v2
```

## Authentication

Currently, the API uses IP-based rate limiting. Authentication tokens may be required in future versions.

## Rate Limits

- Regular endpoints: 200 requests per minute, 5 per second
- Search endpoints: 20 requests per minute, 2 per second
- Search suggestions: 30 requests per minute, 3 per second
- Export endpoints: 5 requests per hour
- Import endpoints: 10 requests per hour
- Quality assessment: 5 requests per hour
- Bulk operations: 20 requests per hour

## Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `language_code` | string | Filter by language code (e.g., "fil", "ceb") |
| `limit` | integer | Maximum number of results to return (default: 50) |
| `offset` | integer | Number of results to skip for pagination (default: 0) |

## Response Format

All API responses are returned in JSON format with the following structure:

```json
{
  "data": { ... },     // Success response data
  "error": "...",      // Error message if applicable
  "count": 123,        // Total count for paginated results
  "results": [ ... ]   // Array of results for collection endpoints
}
```

## Endpoints

### Word Retrieval

#### GET /words/id/{word_id}
Retrieve a word by its ID.

#### GET /words/{word}
Retrieve a word by its lemma.

**Parameters:**
- `include_definitions` (boolean, default: true)
- `include_pronunciations` (boolean, default: true)
- `include_etymologies` (boolean, default: true)
- `include_relations` (boolean, default: true)
- `include_forms` (boolean, default: true)
- `include_templates` (boolean, default: true)
- `include_metadata` (boolean, default: true)

**Example Response:**
```json
{
  "id": 1234,
  "lemma": "halimbawa",
  "language_code": "fil",
  "pos": "n",
  "has_baybayin": true,
  "baybayin_form": "ᜑᜎᜒᜋ᜔ᜊᜏ",
  "definitions": [
    {
      "id": 5678,
      "sense_number": 1,
      "definition": "An example or sample of something.",
      "example": "Mabigyan mo nga ako ng halimbawa.",
      "usage_notes": null,
      "source": "UP Diksiyonaryo",
      "date_created": "2023-05-15T10:30:45Z"
    }
  ],
  "completeness_score": 0.85,
  "date_created": "2023-04-10T14:22:33Z",
  "date_modified": "2023-07-22T09:15:05Z"
}
```

#### GET /random
Get a random word from the dictionary.

**Parameters:**
- `language_code` (string): Filter by language code
- `pos` (string): Filter by part of speech
- `has_etymology` (boolean): Filter for words with etymologies
- `has_baybayin` (boolean): Filter for words with Baybayin form
- `min_definitions` (integer): Minimum number of definitions

### Search

#### GET /search
Search for words based on query text.

**Parameters:**
- `q` (string, required): Search query
- `mode` (string, default: "all"): Search mode ("all", "exact", "prefix", "suffix")
- `sort` (string, default: "relevance"): Sort order
- `order` (string, default: "desc"): Sort direction ("asc", "desc")
- `include_full` (boolean, default: false): Include full word details
- `language` (string): Filter by language code
- `pos` (string): Filter by part of speech
- `has_baybayin` (boolean): Filter for words with Baybayin form
- `limit` (integer, default: 50): Maximum number of results
- `offset` (integer, default: 0): Pagination offset

**Example Request:**
```
GET /api/v2/search?q=bahay&mode=exact&limit=10
```

**Example Response:**
```json
{
  "count": 1,
  "results": [
    {
      "id": 5678,
      "lemma": "bahay",
      "language_code": "fil",
      "pos": "n",
      "has_baybayin": true,
      "baybayin_form": "ᜊᜑᜌ᜔",
      "completeness_score": 0.92,
      "definitions": [
        {
          "id": 9012,
          "sense_number": 1,
          "definition": "A structure for human habitation; house.",
          "example": "Malaki ang bahay nila."
        }
      ]
    }
  ]
}
```

#### GET /search/advanced
Advanced search with additional filtering capabilities.

**Parameters:**
- All parameters from regular search
- `min_completeness` (float): Minimum completeness score (0-1)
- `max_completeness` (float): Maximum completeness score (0-1)
- `date_added_from` (datetime): Filter by creation date
- `date_added_to` (datetime): Filter by creation date
- `date_modified_from` (datetime): Filter by modification date
- `date_modified_to` (datetime): Filter by modification date
- `min_definition_count` (integer): Minimum definition count
- `max_definition_count` (integer): Maximum definition count
- `min_relation_count` (integer): Minimum relation count
- `max_relation_count` (integer): Maximum relation count
- `has_etymology` (boolean): Filter for words with etymologies
- `tags` (array): Filter by tags
- `categories` (array): Filter by categories

**Example Request:**
```
GET /api/v2/search/advanced?q=bahay&min_completeness=0.7&has_etymology=true
```

### Baybayin

#### GET /baybayin/search
Search for words with specific Baybayin characters.

**Parameters:**
- `query` (string, required): Baybayin search query (must contain Baybayin characters)
- `language_code` (string): Filter by language
- `limit` (integer, default: 50): Maximum results
- `offset` (integer, default: 0): Pagination offset

**Example Request:**
```
GET /api/v2/baybayin/search?query=ᜊ&language_code=fil&limit=20
```

**Example Response:**
```json
{
  "count": 125,
  "results": [
    {
      "id": 5678,
      "lemma": "bahay",
      "language_code": "fil",
      "baybayin_form": "ᜊᜑᜌ᜔",
      "pos": "n",
      "completeness_score": 0.92
    },
    {
      "id": 5679,
      "lemma": "basa",
      "language_code": "fil",
      "baybayin_form": "ᜊᜐ",
      "pos": "v",
      "completeness_score": 0.88
    }
  ]
}
```

#### GET /baybayin/statistics
Get detailed statistics about Baybayin usage in the dictionary.

**Example Response:**
```json
{
  "overview": {
    "total_words": 15000,
    "with_baybayin": 3500,
    "percentage": 23.33
  },
  "by_language": [
    {
      "language_code": "fil",
      "total_words": 12000,
      "with_baybayin": 3200,
      "percentage": 26.67
    },
    {
      "language_code": "ceb",
      "total_words": 3000,
      "with_baybayin": 300,
      "percentage": 10.0
    }
  ],
  "character_frequency": {
    "fil": {
      "ᜀ": 956,
      "ᜁ": 450,
      "ᜂ": 325,
      "ᜃ": 780,
      "ᜄ": 625
    }
  },
  "completeness": {
    "with_baybayin": 0.78,
    "without_baybayin": 0.62
  }
}
```

#### POST /baybayin/convert
Convert romanized text to Baybayin script.

**Request Body:**
```json
{
  "text": "Magandang umaga",
  "language_code": "fil"
}
```

**Response:**
```json
{
  "original_text": "Magandang umaga",
  "baybayin_text": "ᜋᜄᜈ᜔ᜇᜅ᜔ ᜂᜋᜄ",
  "conversion_rate": 1.0
}
```

### Word Relationships

#### GET /words/{word}/semantic_network
Get a semantic network of related words.

**Parameters:**
- `depth` (integer, default: 2): Depth of relations to traverse
- `max_nodes` (integer, default: 50): Maximum nodes to return
- `include_definitions` (boolean, default: true): Include definitions
- `relation_types` (array): Filter by relation types

**Example Response:**
```json
{
  "nodes": [
    {
      "id": 5678,
      "lemma": "bahay",
      "language_code": "fil",
      "pos": "n",
      "type": "source"
    },
    {
      "id": 5680,
      "lemma": "tahanan",
      "language_code": "fil",
      "pos": "n",
      "type": "related"
    }
  ],
  "links": [
    {
      "source": 5678,
      "target": 5680,
      "relation_type": "synonym",
      "weight": 1.0
    }
  ]
}
```

#### GET /words/{word}/affixations
Get affixation relationships for a word.

#### GET /words/{word}/etymology
Get etymology information for a word.

#### GET /words/{word_id}/etymology/tree
Get the etymology tree for a word.

#### GET /relationships/types
Get available relationship types.

### Word Details

#### GET /words/{word}/pronunciation
Get pronunciation information for a word.

#### GET /words/{word}/forms
Get word forms (inflections, conjugations).

#### GET /words/{word}/templates
Get word templates.

#### GET /words/{word}/definition_relations
Get definition relations for a word.

### Statistics and Analytics

#### GET /statistics
Get basic dictionary statistics.

**Example Response:**
```json
{
  "total_words": 25000,
  "total_definitions": 42000,
  "words_with_etymology": 8500,
  "words_with_baybayin": 3500,
  "words_by_language": {
    "fil": 20000,
    "ceb": 5000
  },
  "words_by_pos": {
    "n": 12500,
    "v": 7500,
    "adj": 3000,
    "adv": 2000
  },
  "average_completeness": 0.72
}
```

#### GET /statistics/advanced
Get detailed dictionary statistics.

#### GET /statistics/timeseries
Get time-series statistics showing dictionary growth over time.

**Parameters:**
- `start_date` (date): Start date for the time series
- `end_date` (date): End date for the time series
- `interval` (string, default: "month"): Time interval ("day", "week", "month", "year")

#### GET /statistics/language/{language_code}
Get detailed statistics for a specific language.

### Dictionary Export/Import

#### GET /export
Export dictionary data with filtering options.

**Parameters:**
- `language_code` (string): Filter by language
- `pos` (string): Filter by part of speech
- `has_baybayin` (boolean): Filter for words with Baybayin
- `min_completeness` (float): Minimum completeness score
- `format` (string, default: "json"): Export format ("json", "csv", "zip")
- `include_definitions` (boolean, default: true): Include definitions
- `include_relations` (boolean, default: true): Include relations
- `include_etymologies` (boolean, default: true): Include etymologies
- `limit` (integer, default: 5000): Maximum entries to export

#### POST /import
Import dictionary data.

**Request Body:**
- Multipart form data with file upload or JSON data

### Quality Assessment

#### GET /quality_assessment
Analyze dictionary data quality, completeness, and identify issues.

**Parameters:**
- `language_code` (string): Filter by language
- `min_completeness` (float): Minimum completeness score
- `max_completeness` (float): Maximum completeness score
- `issue_severity` (string, default: "all"): Filter by issue severity ("all", "critical", "warning", "info")

**Example Response:**
```json
{
  "overview": {
    "total_words_analyzed": 5000,
    "average_completeness": 0.72,
    "critical_issues": 120,
    "warning_issues": 350,
    "info_issues": 750
  },
  "issues_by_type": {
    "missing_definition": 215,
    "missing_etymology": 420,
    "incomplete_pronunciation": 185,
    "baybayin_errors": 65,
    "relation_errors": 80
  },
  "recommendations": [
    {
      "issue": "missing_definition",
      "count": 215,
      "severity": "critical",
      "recommendation": "Add definitions to words with high usage frequency"
    }
  ]
}
```

### Other Endpoints

#### GET /parts_of_speech
Get available parts of speech.

#### POST /bulk_operations
Perform bulk operations on the dictionary.

## Error Handling

The API uses standard HTTP status codes:

- 200: Success
- 400: Bad Request (invalid parameters)
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 429: Too Many Requests (rate limit exceeded)
- 500: Internal Server Error

Error responses include a detailed message:

```json
{
  "error": "Invalid query parameter",
  "message": "Parameter 'mode' must be one of: all, exact, prefix, suffix"
}
```

## Rate Limit Headers

When rate limits apply, the following headers are included in responses:

- `X-RateLimit-Limit`: Total requests allowed in the period
- `X-RateLimit-Remaining`: Requests remaining in the period
- `X-RateLimit-Reset`: Seconds until the rate limit resets

## SDK and Code Examples

### Python
```python
import requests

API_BASE = "https://api.fil-relex.example.com/api/v2"

def search_word(query, mode="all", limit=10):
    response = requests.get(
        f"{API_BASE}/search",
        params={"q": query, "mode": mode, "limit": limit}
    )
    return response.json()

def convert_to_baybayin(text, language_code="fil"):
    response = requests.post(
        f"{API_BASE}/baybayin/convert",
        json={"text": text, "language_code": language_code}
    )
    return response.json()

def get_word_details(word):
    response = requests.get(f"{API_BASE}/words/{word}")
    return response.json()

def get_baybayin_statistics():
    response = requests.get(f"{API_BASE}/baybayin/statistics")
    return response.json()
```

### JavaScript
```javascript
const API_BASE = "https://api.fil-relex.example.com/api/v2";

async function searchWord(query, mode = "all", limit = 10) {
  const response = await fetch(
    `${API_BASE}/search?q=${query}&mode=${mode}&limit=${limit}`
  );
  return await response.json();
}

async function convertToBaybayin(text, languageCode = "fil") {
  const response = await fetch(`${API_BASE}/baybayin/convert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language_code: languageCode })
  });
  return await response.json();
}

async function getWordDetails(word) {
  const response = await fetch(`${API_BASE}/words/${word}`);
  return await response.json();
}

async function getBaybayinStatistics() {
  const response = await fetch(`${API_BASE}/baybayin/statistics`);
  return await response.json();
}
```

## Versioning

The FilRelex API uses versioning in the URL path (e.g., `/api/v2/`). When breaking changes are introduced, a new version will be created while maintaining the previous version for backward compatibility.

## Support

For API support, contact us at api-support@fil-relex.example.com or open an issue on our GitHub repository. 