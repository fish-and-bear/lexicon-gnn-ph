\
// ... existing code ...
                                                 for syn_idx, syn in enumerate(synonyms_list):
                                                     # KWF synonyms seem to be strings directly
                                                     if isinstance(syn, str) and syn.strip() and syn.strip() != formatted_word:
-                                                        syn_word_clean = syn.strip()
+                                                        # Strip HTML tags before further processing
+                                                        syn_word_raw = syn.strip()
+                                                        syn_word_clean = re.sub(r'<[^>]+>', '', syn_word_raw).strip()
+                                                        if not syn_word_clean or syn_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                         try:
                                                             syn_id = get_or_create_word_id(cur, syn_word_clean, language_code, source_identifier=source_identifier)
                                                             if syn_id:
// ... existing code ...
                                            if isinstance(antonyms_list, list):
                                                 for ant_idx, ant in enumerate(antonyms_list):
                                                     if isinstance(ant, str) and ant.strip() and ant.strip() != formatted_word:
-                                                        ant_word_clean = ant.strip()
+                                                        # Strip HTML tags before further processing
+                                                        ant_word_raw = ant.strip()
+                                                        ant_word_clean = re.sub(r'<[^>]+>', '', ant_word_raw).strip()
+                                                        if not ant_word_clean or ant_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                         try:
                                                             ant_id = get_or_create_word_id(cur, ant_word_clean, language_code, source_identifier=source_identifier)
                                                             if ant_id:
// ... existing code ...
                                                        ref_link = ref.get("link") # KWF has link field

                                                    if isinstance(ref_word, str) and ref_word.strip() and ref_word.strip() != formatted_word:
-                                                        ref_word_clean = ref_word.strip()
+                                                         # Strip HTML tags before further processing
+                                                         ref_word_raw = ref_word.strip()
+                                                         ref_word_clean = re.sub(r'<[^>]+>', '', ref_word_raw).strip()
+                                                         if not ref_word_clean or ref_word_clean == formatted_word: continue # Skip if empty after cleaning or self-reference
                                                          try:
                                                             ref_id = get_or_create_word_id(cur, ref_word_clean, language_code, source_identifier=source_identifier)
                                                             if ref_id:
// ... existing code ...
                                item_metadata = {} # Metadata specific to this related item
                                # KWF format: { "term": "...", "link": "..." } or just string
                                if isinstance(rel_item, dict) and 'term' in rel_item:
-                                   related_term = str(rel_item['term']).strip()
+                                   related_term_raw = str(rel_item['term']).strip()
                                    for meta_key, meta_val in rel_item.items():
                                        if meta_key != 'term': item_metadata[meta_key] = meta_val
                                elif isinstance(rel_item, str):
-                                   related_term = rel_item.strip()
+                                   related_term_raw = rel_item.strip()
+                               else:
+                                   related_term_raw = None # Ensure it's None if neither dict nor str

+                               related_term_clean = None
+                               if related_term_raw:
+                                   # Strip HTML tags before further processing
+                                   related_term_clean = re.sub(r'<[^>]+>', '', related_term_raw).strip()

-                               if related_term and len(related_term) <= 255 and related_term != formatted_word:
+                               if related_term_clean and len(related_term_clean) <= 255 and related_term_clean != formatted_word:
                                     try:
                                         related_id = get_or_create_word_id(
-                                           cur, related_term, language_code=language_code,
+                                           cur, related_term_clean, language_code=language_code,
                                             source_identifier=source_identifier
                                         )
                                         if related_id:
// ... existing code ...
                                             if rel_top_id: stats["relations_added"] += 1
                                             # Handle bidirectional if needed
                                             if rel_type_enum.bidirectional:
-                                                # Check if inverse already added to avoid double counting if structure lists both ways
-                                                # For simplicity here, just add inverse - review if duplicates occur
                                                  insert_relation(cur, related_id, word_id, rel_type_enum, source_identifier, metadata=relation_metadata)


                                     except Exception as e:
-                                       logger.warning(f"Error processing top-level related term '{related_term}' (type '{rel_type_raw}') for '{formatted_word}': {e}")
+                                       logger.warning(f"Error processing top-level related term '{related_term_clean}' (type '{rel_type_raw}') for '{formatted_word}': {e}")
                                         error_types[f"RelationError: {e}"] = error_types.get(f"RelationError: {e}", 0) + 1


// ... existing code ...
