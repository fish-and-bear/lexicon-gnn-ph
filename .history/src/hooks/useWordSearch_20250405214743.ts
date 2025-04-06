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
    async () => {
      const apiResult = await searchWords(query, {
        q: query,
        page,
        per_page: perPage,
        exclude_baybayin: true,
      });
      
      const mappedData: SearchResults = {
        words: (apiResult.words || []).map(wordResult => ({
          id: wordResult.id,
          word: wordResult.lemma
        })),
        total: apiResult.total || 0,
        page: apiResult.page || page,
        perPage: apiResult.perPage || perPage
      };
      
      return mappedData;
    },
    { 
      keepPreviousData: true,
      enabled: query.length > 1
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