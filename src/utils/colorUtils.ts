import * as d3 from 'd3';

/**
 * Determines the appropriate text color (light or dark) for a given background hex color.
 * @param hexColor - The background color in hex format (e.g., "#ff5733").
 * @returns "#111111" (dark) or "#f8f8f8" (light).
 */
export const getTextColorForBackground = (hexColor: string): string => {
  try {
    const color = d3.color(hexColor);
    if (!color) return '#111'; // Default dark text
    const rgb = color.rgb();
    // Calculate luminance using the standard formula
    const luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255;
    // Use a threshold of 0.5 to decide between light and dark text
    return luminance > 0.5 ? '#111111' : '#f8f8f8';
  } catch (e) {
    console.error("Error parsing color for text:", hexColor, e);
    return '#111'; // Fallback to dark text on error
  }
};

/**
 * Maps a detailed relationship type string to a broader category group used for coloring.
 * @param relationship - The specific relationship type (e.g., "derived_from", "synonym").
 * @returns A general group name (e.g., "root", "synonym", "related").
 */
export const mapRelationshipToGroup = (relationship?: string): string => {
    if (!relationship) return 'related'; // Default to 'related' if type is missing
    const relLower = relationship.toLowerCase();
    let group: string;
    
    // Map specific types to broader groups
    switch (relLower) {
      case 'main': group = 'main'; break;
      case 'synonym': group = 'synonym'; break;
      case 'antonym': 
      case 'kasalungat': group = 'antonym'; break;
      case 'related': 
      case 'kaugnay':
      case 'kahulugan':
      case 'similar':
      case 'associated': group = 'related'; break;
      // Add translation types to the 'related' group for color consistency
      case 'has_translation':
      case 'translation_of': group = 'translation'; break;
      case 'variant':
      case 'spelling_variant':
      case 'regional_variant':
      case 'atapat':
      case 'inatapat':
      case 'itapat': group = 'variant'; break;
      // Consolidate root/origin types - SEPARATE root_of and derived_from
      case 'derived': 
      case 'derivative':
      case 'sahod':
      case 'isahod': group = 'derived'; break; // Keep general 'derived' group
      case 'affix': group = 'affix'; break; // ADDED: New group for affix
      case 'root': 
      case 'root_of': group = 'root_of'; break; // Consolidated group for root/root_of
      case 'derived_from': group = 'derived_from'; break; // Specific group for derived_from
      // Taxonomic hierarchy
      case 'hypernym':
      case 'hyponym': group = 'taxonomic'; break;
      // Part-whole relationships
      case 'meronym':
      case 'holonym':
      case 'part_whole':
      case 'component':
      case 'component_of': group = 'part_whole'; break;
      // Etymological relationships
      case 'etymology': group = 'etymology'; break;
      case 'cognate': group = 'cognate'; break;
      // Usage/Comparison - Consolidate under usage_info -> NOW under related
      case 'see_also':
      case 'compare_with':
      case 'usage': 
      // Also map 'other' here if consolidating fully
      case 'other': group = 'related'; break; // Map to related
      // Default fallback group
      default: group = 'related'; 
    }
    
    return group;
};

/**
 * Gets the specific hex color code associated with a relationship group.
 * @param group - The relationship group name (e.g., "root", "synonym").
 * @returns A hex color string (e.g., "#e63946").
 */
export const getNodeColor = (group: string): string => {
    // Colors organized by semantic relationship categories
    switch (group.toLowerCase()) {
      // Core Word
      case "main": return "#003f5c"; // Slightly different deep blue
      
      // Origin Group (Root/Etymology/Cognate)
      case "root_of": return "#c00000"; // Strong Red (was #d00000)
      case "derived_from": return "#FFA500"; // Swapped: Was #8B4513 (SaddleBrown)
      case "etymology": return "#D87093"; // PaleVioletRed (was #c94072)
      case "cognate": return "#8B4513"; // Swapped: Was #FFA500 (Standard Orange)
      
      // Meaning Group (Synonym/Related/Antonym)
      case "synonym": return "#457b9d"; // Muted Slate Blue (was #2a9d8f)
      case "related": return "#80ced6"; // Lighter, softer blue/cyan (was #48cae4)
      case "antonym": return "#6a0dad"; // Distinct Purple (was #023e8a)
      
      // Form Group (Variant)
      case "variant": return "#c34f9a"; // Different purple/magenta (was #7d4fc3)
      
      // Structure Group (Taxonomic/Part-Whole)
      case "taxonomic": return "#588c7e"; // Muted, darker Green/Teal (was #2a9d8f)
      case "part_whole": return "#8c9a58"; // Olive/Khaki Green (was #40916c)
      
      // Derivation Group (Derived/Affix)
      case "derived": return "#f4a261"; // Sandy Brown/Orange (was #fab422)
      case "affix": return "#2a9d8f"; // Teal (New for affix)
      
      // Info Group (Usage) - REMOVED as it maps to related
      // case "usage_info": return "#fca311"; 
      
      // Fallback Color
      default: return "#adb5bd"; // Neutral gray
    }
};

/**
 * Provides a user-friendly label for a relationship type or group.
 * @param type - The specific relationship type (e.g., "derived_from") or group (e.g., "root").
 * @returns An object containing the group name and a display-friendly label.
 */
export const getRelationshipTypeLabel = (type: string): { group: string, label: string } => {
  // Map the specific type to its broader group first
  const group = mapRelationshipToGroup(type);

  // Define labels based on the group name
  switch (group.toLowerCase()) {
    case 'main': return { group: 'Core', label: 'Main Word' };
    case 'root_of': return { group: 'Origin', label: 'Root Of' }; // Consolidated label
    case 'derived_from': return { group: 'Origin', label: 'Derived From' }; // Specific label
    case 'etymology': return { group: 'Origin', label: 'Etymology' };
    case 'cognate': return { group: 'Origin', label: 'Cognate' };
    case 'synonym': return { group: 'Meaning', label: 'Synonym' };
    case 'antonym': return { group: 'Meaning', label: 'Antonym' };
    case 'related': return { group: 'Meaning', label: 'Related' };
    case 'variant': return { group: 'Form', label: 'Variant' };
    case 'taxonomic': return { group: 'Structure', label: 'Taxonomic' };
    case 'part_whole': return { group: 'Structure', label: 'Component/Part' };
    case 'derived': return { group: 'Derivation', label: 'Constructions' };
    case 'affix': return { group: 'Derivation', label: 'Affix' };
    case 'translation': return { group: 'Meaning', label: 'Translation' };
    // case 'usage_info': return { group: 'Info', label: 'Usage / Info' }; // REMOVED consolidated label
    // Fallback: Format the original type if no specific group label is defined
    default: 
      const formattedLabel = type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      return { group: 'Other', label: formattedLabel };
  }
}; 