export interface Definition {
    category: string;
    meaning: string;
    example: string;
    opposites?: string[];
    relatedWords: string[];
  }
  
  export interface WordData {
    type: string;
    pronunciationUK: string;
    pronunciationUS: string;
    definitions: Definition[];
  }
  
  export const mockDictionary: { [key: string]: WordData } = {
    'Abstract': {
      type: 'adjective',
      pronunciationUK: '/æbˈstrækt/',
      pronunciationUS: '/æbˈstrækt/',
      definitions: [
        {
          category: 'General',
          meaning: 'Existing as an idea, feeling, or quality, not as a material object:',
          example: 'Truth and beauty are abstract concepts',
          opposites: ['concrete'],
          relatedWords: ['conceptual', 'notional', 'theoretical', 'virtual', 'ideal'],
        },
        {
          category: 'Art',
          meaning: 'Used to refer to a type of painting, drawing, or sculpture that uses shapes, lines, and colour in a way that doesn\'t try to represent the appearance of people or things:',
          example: 'Abstract art',
          relatedWords: ['non-representational', 'non-objective', 'formalist'],
        },
      ],
    },
    // Add more words here...
  };