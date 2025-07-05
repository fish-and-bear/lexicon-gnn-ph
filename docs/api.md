# API Documentation

This document describes the REST API endpoints provided by the Philippine Lexicon GNN Toolkit backend.

## Base URL

- Local: `http://localhost:5000/api/v2/`
- Production: [https://explorer.hapinas.net/](https://explorer.hapinas.net/)

## Authentication

- The API is public for read-only endpoints. For write/admin endpoints (if enabled), use API keys or JWT tokens as described in the backend configuration (see `backend/routes.py`).

## Endpoints

### `GET /search`
Search for words in the lexicon.

**Parameters:**
- `q` (string, required): Query word
- `language` (string, optional): Language code (e.g., `tl`, `ceb`)
- `limit` (int, optional): Max results (default: 20)

**Example Request:**
```
curl "http://localhost:5000/api/v2/search?q=kain&language=tl"
```

**Example Response:**
```json
{
  "results": [
    {"id": 123, "lemma": "kain", "language": "tl", "definition": "to eat"},
    ...
  ]
}
```

**Error Codes:**
- 400: Missing required parameters
- 500: Internal server error

---

### `GET /words/<id>`
Get details for a specific word.

**Example Request:**
```
curl "http://localhost:5000/api/v2/words/123"
```

**Example Response:**
```json
{
  "id": 123,
  "lemma": "kain",
  "language": "tl",
  "definition": "to eat",
  "etymology": "from Proto-Austronesian *kaen"
}
```

**Error Codes:**
- 404: Word not found

---

### `GET /semantic_network`
Retrieve the semantic network for a word.

**Parameters:**
- `word` (string, required): Query word
- `depth` (int, optional): Graph depth (default: 2)

**Example Request:**
```
curl "http://localhost:5000/api/v2/semantic_network?word=kain&depth=2"
```

**Example Response:**
```json
{
  "nodes": [...],
  "edges": [...]
}
```

---

### `GET /etymology/tree`
Get the etymology tree for a word.

**Parameters:**
- `word` (string, required): Query word

**Example Request:**
```
curl "http://localhost:5000/api/v2/etymology/tree?word=kain"
```

**Example Response:**
```json
{
  "tree": {...}
}
```

---

## Error Handling

All endpoints return standard HTTP status codes and JSON error messages. See above for endpoint-specific codes.

## Usage Notes

- For bulk queries or advanced features, see the backend code or open an issue for support.
- Rate limiting may apply in production deployments.
- For write/admin endpoints, see backend documentation (not enabled in public demo).

For more details, see the source code in `backend/routes.py` or open an issue for clarification. 