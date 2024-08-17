// Define the structure for a single meaning and its source
export interface Meaning {
    meaning: string;
}

export interface Source {
    source: string;
}

// Define the structure for a single definition, including part of speech, meanings, and sources
export interface Definition {
    part_of_speech: string;
    meanings: string[]; // An array of meanings
    sources: string[];  // An array of sources corresponding to the meanings
}

// Define the structure for word information returned by the API
export interface WordInfo {
    word: string;
    pronunciation?: string;
    etymology?: string;
    language_codes?: string[];  // Assuming language codes are an array of strings
    definitions?: Definition[];  // An array of definitions
    derivatives?: Record<string, string>;  // Assuming derivatives is a dictionary of strings
    associated_words?: string[];  // An array of associated words
    root_words?: string[];  // An array of root words
    root_word?: string;  // The root word if available
}

// Define the structure for the word network, mapping words to their info
export interface WordNetwork {
    [key: string]: WordInfo;
}
