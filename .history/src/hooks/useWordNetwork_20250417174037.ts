import { useQuery } from '@tanstack/react-query';
import { fetchWordNetwork } from '../api/wordApi';
import { WordNetwork } from '../types';

export interface NetworkOptions {
  depth?: number;
  breadth?: number;
  include_affixes?: boolean;
  include_etymology?: boolean;
  cluster_threshold?: number;
}

export function useWordNetwork(word: string | undefined, options: NetworkOptions) {
  return useQuery<WordNetwork>({
    queryKey: ['wordNetwork', word, options],
    queryFn: () => fetchWordNetwork(word!, options),
    enabled: Boolean(word),
    staleTime: 1000 * 60 * 60, // Cache for 1 hour
  });
} 