import { useState, useCallback } from 'react';
import { useQuery } from 'react-query';
import { searchWords } from '../api/wordApi';
import { SearchOptions, SearchResults, SearchResultItem } from '../types';

// Define the specific return type for this hook
interface UseWordSearchReturn {
  words: { id: number; word: string }[];
  total: number;
  page: number;
  perPage: number;
}

export function useWordSearch(initialQuery: string = '') {
  const [query, setQuery] = useState(initialQuery);
  const [page, setPage] = useState(1);
  const perPage = 20;

  const { data, isLoading, error } = useQuery<UseWordSearchReturn, Error>(
    ['wordSearch', query, page],
    async () => {
      const apiResult = await searchWords(query, {
        page,
        per_page: perPage,
        exclude_baybayin: true,
      });
      
      const mappedData: UseWordSearchReturn = {
        words: (apiResult.results || []).map((wordResult: SearchResultItem) => ({
          id: wordResult.word_id,
          word: wordResult.lemma
        })),
        total: apiResult.total || 0,
        page: apiResult.page || page,
        perPage: apiResult.per_page || perPage
      };
      
      return mappedData;
    },
    { 
      keepPreviousData: true,
      enabled: query.length > 1
    }
  );

  // Update the type annotation for the destructured data variable
  const typedData: UseWordSearchReturn | undefined = data;

  const handleSearch = useCallback((newQuery: string) => {
    setQuery(newQuery);
    setPage(1);
  }, []);

  return {
    query,
    setQuery: handleSearch,
    page,
    setPage,
    data: typedData, // Return the correctly typed data
    isLoading,
    error,
  };
}