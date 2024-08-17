import { WordData } from '../types/wordTypes';

export const mockWordData: { [key: string]: WordData } = {
  "abstract": {
    word: "abstract",
    type: "adjective",
    definition: "Existing in thought or as an idea but not having a physical or concrete existence.",
    etymology: "From Latin abstractus, past participle of abstrahere 'draw away'",
    relatedWords: ["abstraction", "abstractly", "abstractness"]
  },
  "abstraction": {
    word: "abstraction",
    type: "noun",
    definition: "The quality of dealing with ideas rather than events.",
    etymology: "Late Middle English: from Latin abstractio(n-), from the verb abstrahere 'draw away'",
    relatedWords: ["abstract", "abstractly", "abstractness"]
  },
  "abstractly": {
    word: "abstractly",
    type: "adverb",
    definition: "In a way that is based on general ideas or principles rather than specific examples or real events.",
    etymology: "From abstract + -ly",
    relatedWords: ["abstract", "abstraction", "abstractness"]
  },
  "abstractness": {
    word: "abstractness",
    type: "noun",
    definition: "The quality of being abstract.",
    etymology: "From abstract + -ness",
    relatedWords: ["abstract", "abstraction", "abstractly"]
  }
};