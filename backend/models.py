import os
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Table, JSON, ARRAY
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.associationproxy import association_proxy
from dotenv import load_dotenv
from database import Base  # This is important

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# Association Tables
word_language_table = Table(
    'word_languages', Base.metadata,
    Column('word_id', Integer, ForeignKey('words.id'), primary_key=True),
    Column('language_id', Integer, ForeignKey('languages.id'), primary_key=True)
)

synonym_table = Table(
    'synonyms', Base.metadata,
    Column('word_id', Integer, ForeignKey('words.id'), primary_key=True),
    Column('synonym_id', Integer, ForeignKey('words.id'), primary_key=True)
)

antonym_table = Table(
    'antonyms', Base.metadata,
    Column('word_id', Integer, ForeignKey('words.id'), primary_key=True),
    Column('antonym_id', Integer, ForeignKey('words.id'), primary_key=True)
)

related_term_table = Table(
    'related_terms', Base.metadata,
    Column('word_id', Integer, ForeignKey('words.id'), primary_key=True),
    Column('related_term_id', Integer, ForeignKey('words.id'), primary_key=True)
)

class Word(Base):
    __tablename__ = 'words'
    
    id = Column(Integer, primary_key=True)
    word = Column(String, unique=True, nullable=False)
    pronunciation = Column(Text)
    root_word = Column(String, ForeignKey('words.word'))
    audio_pronunciation = Column(ARRAY(String))
    tags = Column(ARRAY(String))
    kaikki_etymology = Column(Text)
    variant = Column(String)

    etymologies = relationship("Etymology", back_populates="word", cascade="all, delete-orphan")
    definitions = relationship("Definition", back_populates="word", cascade="all, delete-orphan")
    forms = relationship("Form", back_populates="word", cascade="all, delete-orphan")
    head_templates = relationship("HeadTemplate", back_populates="word", cascade="all, delete-orphan")
    derivatives = relationship("Derivative", back_populates="word", cascade="all, delete-orphan")
    examples = relationship("Example", back_populates="word", cascade="all, delete-orphan")
    associated_words = relationship("AssociatedWord", back_populates="word", cascade="all, delete-orphan")
    alternate_forms = relationship("AlternateForm", back_populates="word", cascade="all, delete-orphan")
    inflections = relationship("Inflection", back_populates="word", cascade="all, delete-orphan")

    languages = relationship("Language", secondary=word_language_table, back_populates="words")
    synonyms = relationship("Word", secondary=synonym_table,
                            primaryjoin=(id==synonym_table.c.word_id),
                            secondaryjoin=(id==synonym_table.c.synonym_id))
    antonyms = relationship("Word", secondary=antonym_table,
                            primaryjoin=(id==antonym_table.c.word_id),
                            secondaryjoin=(id==antonym_table.c.antonym_id))
    related_terms = relationship("Word", secondary=related_term_table,
                                 primaryjoin=(id==related_term_table.c.word_id),
                                 secondaryjoin=(id==related_term_table.c.related_term_id))

    hypernyms = relationship("Hypernym", back_populates="word", cascade="all, delete-orphan")
    hyponyms = relationship("Hyponym", back_populates="word", cascade="all, delete-orphan")
    meronyms = relationship("Meronym", back_populates="word", cascade="all, delete-orphan")
    holonyms = relationship("Holonym", back_populates="word", cascade="all, delete-orphan")

class Etymology(Base):
    __tablename__ = 'etymologies'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    etymology_text = Column(Text)

    word = relationship("Word", back_populates="etymologies")
    components = relationship("EtymologyComponent", back_populates="etymology", cascade="all, delete-orphan")

class EtymologyComponent(Base):
    __tablename__ = 'etymology_components'

    id = Column(Integer, primary_key=True)
    etymology_id = Column(Integer, ForeignKey('etymologies.id'), nullable=False)
    component = Column(String, nullable=False)
    order = Column(Integer, nullable=False)

    etymology = relationship("Etymology", back_populates="components")

class Definition(Base):
    __tablename__ = 'definitions'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    part_of_speech = Column(String, nullable=False)
    usage_notes = Column(ARRAY(Text))
    tags = Column(ARRAY(String))
    
    word = relationship("Word", back_populates="definitions")
    meanings = relationship("Meaning", back_populates="definition", cascade="all, delete-orphan")
    examples = relationship("Example", back_populates="definition", cascade="all, delete-orphan")

    def matches(self, new_data):
        return self.part_of_speech == new_data.get('part_of_speech')

    def update(self, new_data):
        self.part_of_speech = new_data.get('part_of_speech', self.part_of_speech)
        self.usage_notes = new_data.get('usage_notes', self.usage_notes)
        self.tags = new_data.get('tags', self.tags)
        # Update meanings if necessary

class Meaning(Base):
    __tablename__ = 'meanings'
    
    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id'), nullable=False)
    source_id = Column(Integer, ForeignKey('sources.id'), nullable=False)
    meaning = Column(Text, nullable=False)
    
    definition = relationship("Definition", back_populates="meanings")
    source = relationship("Source", back_populates="meanings")

class Source(Base):
    __tablename__ = 'sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String, unique=True, nullable=False)
    
    meanings = relationship("Meaning", back_populates="source")

class Form(Base):
    __tablename__ = 'forms'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    form = Column(String, nullable=False)
    tags = Column(ARRAY(String))
    
    word = relationship("Word", back_populates="forms")

class HeadTemplate(Base):
    __tablename__ = 'head_templates'
    
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    template_name = Column(String, nullable=False)
    args = Column(JSON)
    expansion = Column(Text)
    
    word = relationship("Word", back_populates="head_templates")

class Language(Base):
    __tablename__ = 'languages'

    id = Column(Integer, primary_key=True)
    code = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=True)

    words = relationship("Word", secondary=word_language_table, back_populates="languages")

class Derivative(Base):
    __tablename__ = 'derivatives'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    derivative = Column(String, nullable=False)

    word = relationship("Word", back_populates="derivatives")

class Example(Base):
    __tablename__ = 'examples'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    definition_id = Column(Integer, ForeignKey('definitions.id'), nullable=True)
    example = Column(Text, nullable=False)

    word = relationship("Word", back_populates="examples")
    definition = relationship("Definition", back_populates="examples")

class AssociatedWord(Base):
    __tablename__ = 'associated_words'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    associated_word = Column(String, nullable=False)

    word = relationship("Word", back_populates="associated_words")

class AlternateForm(Base):
    __tablename__ = 'alternate_forms'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    alternate_form = Column(String, nullable=False)

    word = relationship("Word", back_populates="alternate_forms")

class Inflection(Base):
    __tablename__ = 'inflections'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    name = Column(String, nullable=False)
    args = Column(JSON)

    word = relationship("Word", back_populates="inflections")

class Hypernym(Base):
    __tablename__ = 'hypernyms'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    hypernym = Column(String, nullable=False)

    word = relationship("Word", back_populates="hypernyms")

class Hyponym(Base):
    __tablename__ = 'hyponyms'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    hyponym = Column(String, nullable=False)

    word = relationship("Word", back_populates="hyponyms")

class Meronym(Base):
    __tablename__ = 'meronyms'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    meronym = Column(String, nullable=False)

    word = relationship("Word", back_populates="meronyms")

class Holonym(Base):
    __tablename__ = 'holonyms'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'), nullable=False)
    holonym = Column(String, nullable=False)

    word = relationship("Word", back_populates="holonyms")

# Database setup functions
def get_engine():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in the environment variables")
    return create_engine(DATABASE_URL)

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)

Session = sessionmaker(bind=get_engine())