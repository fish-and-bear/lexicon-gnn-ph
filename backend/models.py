from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
import json

class Word(Base):
    __tablename__ = 'words'

    id = Column(Integer, primary_key=True)
    word = Column(String(255), unique=True, nullable=False)
    pronunciation = Column(Text)
    etymology = Column(Text)
    language_codes = Column(Text)  # New column to store language codes
    derivatives = Column(Text)
    root_word = Column(Text)
    definitions = relationship('Definition', back_populates='word')
    associated_words = relationship('AssociatedWord', back_populates='word')

    @property
    def derivatives_dict(self):
        return json.loads(self.derivatives) if self.derivatives else {}

    @derivatives_dict.setter
    def derivatives_dict(self, value):
        self.derivatives = json.dumps(value) if value else None

class Definition(Base):
    __tablename__ = 'definitions'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    part_of_speech = Column(Text)
    word = relationship('Word', back_populates='definitions')
    meanings = relationship('Meaning', back_populates='definition')
    sources = relationship('Source', back_populates='definition')

class Meaning(Base):
    __tablename__ = 'meanings'

    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id'))
    meaning = Column(Text)
    definition = relationship('Definition', back_populates='meanings')

class Source(Base):
    __tablename__ = 'sources'

    id = Column(Integer, primary_key=True)
    definition_id = Column(Integer, ForeignKey('definitions.id'))
    source = Column(Text)
    definition = relationship('Definition', back_populates='sources')

class AssociatedWord(Base):
    __tablename__ = 'associated_words'

    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    associated_word = Column(Text)
    word = relationship('Word', back_populates='associated_words')