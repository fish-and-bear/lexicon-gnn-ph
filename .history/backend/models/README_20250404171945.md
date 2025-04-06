# Model Development Guidelines

## SQLAlchemy Naming Conflicts

SQLAlchemy has several reserved attribute names that should not be used as column attributes in your models:

### Critical Reserved Names

- `metadata` - Used for the MetaData instance in declarative base classes
- `query` - Reserved for query property
- `query_class` - Reserved for query class specification
- `__table__` - Used to refer to the SQLAlchemy Table object
- `__tablename__` - Used to specify the database table name
- `__mapper__` - Used to refer to the SQLAlchemy Mapper object

### How to Handle 'metadata' Column Names

If you need a column called 'metadata' in your database, use this pattern:

```python
# Map a different attribute name to the 'metadata' column
some_other_name = db.Column('metadata', db.JSON)

# Optionally add properties to expose it as 'metadata' in Python
@property
def metadata(self):
    return self.some_other_name

@metadata.setter
def metadata(self, value):
    self.some_other_name = value

# In your to_dict method
def to_dict(self):
    return {
        # ...
        'metadata': self.some_other_name,  # Expose as 'metadata' in API
        # ...
    }
```

### Real Examples in this Codebase

1. **Pronunciation Model**: Uses `pronunciation_metadata` as the attribute name but maps to 'metadata' column:
   ```python
   pronunciation_metadata = db.Column('metadata', db.JSON)
   ```

2. **Relation Model**: Uses `extra_data` as the attribute name but maps to 'metadata' column:
   ```python
   extra_data = db.Column('metadata', db.JSON, default=dict, server_default='{}')
   ```

Remember to handle this consistently in routes and all parts of the code that interact with these models. 