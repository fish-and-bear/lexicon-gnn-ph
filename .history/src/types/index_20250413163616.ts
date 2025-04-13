// Import RelatedWord from the main types file
import { RelatedWord } from '../types';

// Add this interface modification to support string indexing on Relation objects
export interface Relation {
  id: number;
  relation_type: string;
  source_word?: RelatedWord;
  target_word?: RelatedWord;
  // Allow string indexing for dynamic property access
  [key: string]: any;
} 