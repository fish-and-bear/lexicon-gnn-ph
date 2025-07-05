# API Documentation

This document describes the REST API endpoints provided by the Philippine Lexicon GNN Toolkit backend.

## Base URL

- Local: `http://localhost:5000/api/v2/`
- Production: See [https://explorer.hapinas.net/](https://explorer.hapinas.net/)

## Endpoints

### `GET /search`
Search for words in the lexicon.

**Parameters:**
- `q` (string): Query word
- `language` (string, optional): Language code (e.g., `tl`, `ceb`)

**Example:**
```
GET /api/v2/search?q=kain&language=tl
```

---

### `GET /words/<id>`
Get details for a specific word.

**Example:**
```
GET /api/v2/words/123
```

---

### `GET /semantic_network`
Retrieve the semantic network for a word.

**Parameters:**
- `word` (string): Query word
- `depth` (int, optional): Graph depth

---

### `GET /etymology/tree`
Get the etymology tree for a word.

**Parameters:**
- `word` (string): Query word

---

## Error Handling

All endpoints return standard HTTP status codes and JSON error messages.

---

For more details, see the source code in `backend/routes.py` or open an issue for clarification. 