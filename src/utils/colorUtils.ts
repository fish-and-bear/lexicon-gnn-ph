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
      case 'variant':
      case 'spelling_variant':
      case 'regional_variant':
      case 'atapat':
      case 'inatapat':
      case 'itapat': group = 'variant'; break;
      // Consolidate root/origin types
      case 'derived': 
      case 'derived_from':
      case 'sahod':
      case 'root':
      case 'root_of':
      case 'isahod': group = 'root'; break;
      // Consolidate derivation types (can use 'derived' or 'root' group color)
      case 'affix':
      case 'derivative': group = 'derived'; break; 
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
      // Usage/Comparison
      case 'see_also':
      case 'compare_with':
      case 'usage': group = 'usage'; break;
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
      case "main": return "#0e4a86"; // Deep blue
      
      // Origin Group (Root/Etymology/Cognate)
      case "root": return "#e63946"; // Bright red
      case "etymology": return "#d00000"; // Dark red
      case "cognate": return "#ff5c39"; // Light orange
      
      // Meaning Group (Synonym/Related/Antonym)
      case "synonym": return "#457b9d"; // Medium blue
      case "related": return "#48cae4"; // Light blue
      case "antonym": return "#023e8a"; // Dark blue
      
      // Form Group (Variant)
      case "variant": return "#7d4fc3"; // Medium purple
      
      // Hierarchy Group (Taxonomic/Part-Whole)
      case "taxonomic": return "#2a9d8f"; // Teal
      case "part_whole": return "#40916c"; // Forest green
      
      // Derivation Group (Derived/Affix)
      case "derived": return "#2a9d8f"; // Teal (Same as Taxonomic for visual consistency)
      
      // Info Group (Usage)
      case "usage": return "#fcbf49"; // Gold
      
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
    case 'root': return { group: 'Origin', label: 'Root/Origin' };
    case 'etymology': return { group: 'Origin', label: 'Etymology' };
    case 'cognate': return { group: 'Origin', label: 'Cognate' };
    case 'synonym': return { group: 'Meaning', label: 'Synonym' };
    case 'antonym': return { group: 'Meaning', label: 'Antonym' };
    case 'related': return { group: 'Meaning', label: 'Related' };
    case 'variant': return { group: 'Form', label: 'Variant' };
    case 'taxonomic': return { group: 'Structure', label: 'Taxonomic' };
    case 'part_whole': return { group: 'Structure', label: 'Component/Part' };
    case 'derived': return { group: 'Derivation', label: 'Derived' };
    case 'usage': return { group: 'Info', label: 'Usage Note' };
    // Fallback: Format the original type if no specific group label is defined
    default: 
      const formattedLabel = type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
      return { group: 'Other', label: formattedLabel };
  }
}; 