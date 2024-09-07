import { useState, useCallback } from 'react';
import { useQuery } from 'react-query';
import { searchWords } from '../api/wordApi';

interface SearchResult {
  word: string;
  id: number;
}

interface SearchResults {
  words: SearchResult[];
  total: number;
}

interface SearchOptions {
  page: number;
  per_page: number;
  fuzzy: boolean;
  filter: string; // Added filter property
}

export function useWordSearch(initialQuery: string = '') {
  const [query, setQuery] = useState(initialQuery);
  const [page, setPage] = useState(1);
  const perPage = 20;

  const { data, isLoading, error } = useQuery<SearchResults, Error>(
    ['wordSearch', query, page],
    () => searchWords(query, { 
      page, 
      per_page: perPage, 
      filter: 'is_real_word:true' 
    }),
    { keepPreviousData: true }
  );

  const handleSearch = useCallback((newQuery: string) => {
    setQuery(newQuery);
    setPage(1);
  }, []);

  return {
    query,
    setQuery: handleSearch,
    page,
    setPage,
    data,
    isLoading,
    error,
  };
}