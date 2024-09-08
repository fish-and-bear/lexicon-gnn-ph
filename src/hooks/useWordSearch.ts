import { useState, useCallback } from 'react';
import { useQuery } from 'react-query';
import { searchWords } from '../api/wordApi';
import { SearchOptions, SearchResult } from '../types';

interface SearchResults {
  words: { id: number; word: string; }[];
  total: number;
  page: number;
  perPage: number;
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
      exclude_baybayin: true,
      is_real_word: true
    }),
    { 
      keepPreviousData: true,
      enabled: query.length > 0 // Only run query when there's input
    }
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