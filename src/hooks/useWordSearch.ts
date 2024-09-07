import { useState, useCallback } from 'react';
import { useQuery } from 'react-query';
import { searchWords } from '../api/wordApi';
import { SearchOptions, SearchResult } from '../types';

interface SearchResults {
  words: SearchResult[];
  total: number;
}

export function useWordSearch(initialQuery: string = '') {
  const [query, setQuery] = useState(initialQuery);
  const [page, setPage] = useState(1);
  const perPage = 20;

  const { data, isLoading, error } = useQuery<SearchResult, Error>(
    ['wordSearch', query, page],
    () => searchWords(query, { 
      page, 
      per_page: perPage,
      filter: '',
      exclude_baybayin: true
    } as SearchOptions),
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