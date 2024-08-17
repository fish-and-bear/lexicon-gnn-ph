export interface WordData {
    word: string;
    type: string;
    definition: string;
    etymology: string;
    relatedWords: string[];
  }
  
  export interface WordNode extends d3.SimulationNodeDatum {
    id: string;
    group: number;
  }
  
  export interface WordLink extends d3.SimulationLinkDatum<WordNode> {
    source: string;
    target: string;
  }