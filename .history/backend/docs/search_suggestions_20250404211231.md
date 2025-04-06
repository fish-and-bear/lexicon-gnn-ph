# Efficient Search Suggestions System

This document describes the efficient search suggestions system implemented for the Fil-Relex backend.

## Features

1. **Optimized Database Queries**
   - Single efficiently indexed query instead of multiple queries
   - Uses PostgreSQL-specific optimizations (trigrams, text pattern operators)
   - Multi-strategy approach with prefix, similarity, and text search

2. **Database Indexes**
   ```sql
   -- For prefix matching (very fast for startswith queries)
   CREATE INDEX words_normalized_lemma_pattern_idx ON words (normalized_lemma text_pattern_ops);
   
   -- For trigram matching (flexible similarity search)
   CREATE EXTENSION pg_trgm;
   CREATE INDEX words_normalized_lemma_trgm_idx ON words USING gin (normalized_lemma gin_trgm_ops);
   
   -- For definition text search
   CREATE INDEX definitions_text_search_idx ON definitions USING gin (to_tsvector('english', definition_text));
   ```

3. **Asynchronous Logging**
   - Background thread-based logging system
   - Batched database writes for efficiency
   - Never impacts user experience with logging delays

4. **Result Caching**
   - Uses `@cached_query` decorator to cache frequent suggestion results
   - Short cache lifetime (60 seconds) balances freshness with performance
   - Dramatically reduces database load for popular partial queries

5. **Optimized Payload**
   - Only includes fields that have values
   - Keeps the response size minimal for faster network transfer

6. **Failsafe Design**
   - Graceful degradation with simpler fallback queries
   - Ultra-basic last resort query when all else fails
   - Will never return an error to the client

7. **Popular Suggestions**
   - Tracks word selection using materialized view for fast lookup
   - Refreshes periodically for up-to-date popular suggestions
   - Prioritizes root words and commonly searched items

## API Endpoints

### 1. Search Suggestions

```
GET /api/search/suggestions
```

**Query Parameters:**
- `q`: Search query (required, minimum 2 characters)
- `language`: Filter by language code (optional)
- `limit`: Maximum number of suggestions (optional, default 10, max 20)

**Example Response:**
```json
{
  "suggestions": [
    {
      "id": 123,
      "text": "tulog",
      "type": "prefix_match",
      "confidence": 0.95,
      "language": "tl",
      "has_baybayin": true,
      "baybayin_form": "ᜆᜓᜎᜓᜄ᜔"
    },
    {
      "id": 456,
      "text": "tulungan",
      "type": "popular_match",
      "confidence": 0.9,
      "language": "tl"
    },
    {
      "id": 789,
      "text": "tulong",
      "type": "spelling_suggestion",
      "confidence": 0.82,
      "language": "tl",
      "definition_preview": "aid, assistance, help"
    }
  ]
}
```

### 2. Track Search Selection

```
POST /api/search/track-selection
```

**Request Body:**
```json
{
  "query": "original search query",
  "selected_id": 123,   // Word ID that was selected (optional if selected_text provided)
  "selected_text": "selected word" // Text that was selected (optional if selected_id provided)
}
```

**Example Response:**
```json
{
  "status": "success"
}
```

## Implementation Details

### Database Schema

The system adds the following database objects:

1. **search_logs Table**
   - Stores user search queries and selections
   - Used for analytics and improving suggestion quality

2. **popular_words Materialized View**
   - Aggregates selection data for fast lookup
   - Refreshed periodically to stay current

### Background Processing

The system uses a background thread to:

1. Process search log entries in batches
2. Refresh the popular_words materialized view periodically
3. Handle any logging errors without impacting user experience

## Performance Considerations

1. **Query Complexity Limits**
   - Trigram similarity searches are limited to 3 results to control performance
   - Definition text searches are limited to 3 results
   - Prefix matches get the majority of the result slots

2. **Cache Management**
   - Short cache lifetime (60 seconds) keeps suggestions fresh
   - Cache reduces database load, especially for common prefixes

3. **Payload Optimization**
   - Optional fields (baybayin, definition preview, etc.) only included when available
   - Reduces network transfer size and improves client performance

## Installation and Usage

1. Run the migration script:
   ```
   python -m backend.migrations.add_search_suggestions
   ```

2. Restart the API server to initialize background tasks

3. Use the provided endpoints in your frontend application

## Frontend Integration Example

```javascript
// Search suggestions
async function getSearchSuggestions(query, language = null) {
  if (!query || query.length < 2) return [];
  
  const params = new URLSearchParams({
    q: query
  });
  
  if (language) {
    params.append('language', language);
  }
  
  try {
    const response = await fetch(`/api/search/suggestions?${params.toString()}`);
    const data = await response.json();
    return data.suggestions || [];
  } catch (error) {
    console.error('Error fetching suggestions:', error);
    return [];
  }
}

// Track user selection to improve future suggestions
async function trackSearchSelection(query, selectedId, selectedText) {
  try {
    await fetch('/api/search/track-selection', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query,
        selected_id: selectedId,
        selected_text: selectedText
      })
    });
  } catch (error) {
    // Silent failure - don't affect user experience
    console.error('Error tracking selection:', error);
  }
}
``` 