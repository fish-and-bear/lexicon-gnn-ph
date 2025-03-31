import pytest
from backend.models import Word, Definition, Etymology, PartOfSpeech

def test_get_words(client, db_session):
    # Create test data
    word = Word(
        lemma="test",
        normalized_lemma="test",
        language_code="tl"
    )
    db_session.add(word)
    db_session.commit()

    # Test API endpoint
    response = client.get('/api/v1/words')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['words']) == 1
    assert data['words'][0]['word'] == 'test'

def test_get_word_details(client, db_session):
    # Create test data with relationships
    word = Word(
        lemma="test",
        normalized_lemma="test",
        language_code="tl"
    )
    pos = PartOfSpeech(
        code="n",
        name_en="noun",
        name_tl="pangngalan",
        description="A word that refers to a person, place, thing, or idea"
    )
    definition = Definition(
        word=word,
        definition_text="A test definition",
        part_of_speech=pos,
        sources="test_source"
    )
    etymology = Etymology(
        word=word,
        etymology_text="From test origin",
        sources="test_source"
    )
    db_session.add_all([word, pos, definition, etymology])
    db_session.commit()

    # Test API endpoint
    response = client.get('/api/v1/words/test')
    assert response.status_code == 200
    data = response.get_json()
    assert data['data']['word'] == 'test'
    assert len(data['data']['definitions']) == 1
    assert len(data['data']['etymologies']) == 1

def test_search_words(client, db_session):
    # Create test data
    words = [
        Word(lemma="test1", normalized_lemma="test1", language_code="tl"),
        Word(lemma="test2", normalized_lemma="test2", language_code="tl"),
        Word(lemma="other", normalized_lemma="other", language_code="tl")
    ]
    db_session.add_all(words)
    db_session.commit()

    # Test search endpoint
    response = client.get('/api/v1/search?q=test')
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['results']) == 2

def test_word_network(client, db_session):
    # Create test data with relationships
    word1 = Word(lemma="test1", normalized_lemma="test1", language_code="tl")
    word2 = Word(lemma="test2", normalized_lemma="test2", language_code="tl")
    db_session.add_all([word1, word2])
    db_session.commit()

    # Test network endpoint
    response = client.get('/api/v1/word_network/test1')
    assert response.status_code == 200
    data = response.get_json()
    assert 'test1' in data

def test_etymology(client, db_session):
    # Create test data
    word = Word(lemma="test", normalized_lemma="test", language_code="tl")
    etymology = Etymology(
        word=word,
        etymology_text="From test origin",
        normalized_components="test",
        language_codes="en",
        sources="test_source"
    )
    db_session.add_all([word, etymology])
    db_session.commit()

    # Test etymology endpoint
    response = client.get('/api/v1/etymology/test')
    assert response.status_code == 200
    data = response.get_json()
    assert data['word'] == 'test'
    assert len(data['etymologies']) == 1

def test_bulk_words(client, db_session):
    # Create test data
    words = [
        Word(lemma="test1", normalized_lemma="test1", language_code="tl"),
        Word(lemma="test2", normalized_lemma="test2", language_code="tl")
    ]
    db_session.add_all(words)
    db_session.commit()

    # Test bulk endpoint
    response = client.post('/api/v1/bulk_words', json={
        "words": ["test1", "test2"]
    })
    assert response.status_code == 200
    data = response.get_json()
    assert len(data['words']) == 2 