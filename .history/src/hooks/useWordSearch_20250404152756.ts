import { useState, useCallback } from 'react';
import { useQuery } from 'react-query';
import { searchWords } from '../api/wordApi';
import { SearchOptions, SearchResult } from '../types';

export function useWordSearch(initialQuery: string = '', initialLimit: number = 20) {
  const [query, setQuery] = useState(initialQuery);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(initialLimit);

  const { data, isLoading, error } = useQuery<SearchResult, Error>(
    ['wordSearch', query, limit, offset],
    async () => {
      const apiResult = await searchWords({
        q: query,
        limit: limit,
        offset: offset,
      });
      
      return apiResult;
    },
    { 
      keepPreviousData: true,
      enabled: query.trim().length > 0 
    }
  );

  const handleSearch = useCallback((newQuery: string) => {
    setQuery(newQuery);
    setOffset(0);
  }, []);

  const loadNextPage = useCallback(() => {
    if (data && (offset + limit < data.total)) {
      setOffset(prevOffset => prevOffset + limit);
    }
  }, [data, offset, limit]);

  const loadPreviousPage = useCallback(() => {
    setOffset(prevOffset => Math.max(0, prevOffset - limit));
  }, [limit]);

  return {
    query,
    setQuery: handleSearch,
    offset,
    setOffset,
    limit,
    setLimit,
    loadNextPage,
    loadPreviousPage,
    data,
    isLoading,
    error,
  };
}