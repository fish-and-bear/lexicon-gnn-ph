# CSS Architecture

This project uses a simplified utility-first CSS architecture with CSS custom properties (variables).

## Directory Structure

```
styles/
├── index.css                  # Main stylesheet with base styling and imports
├── utilities/                 # Core utility files
│   ├── _reset.css             # CSS reset/normalization
│   ├── _variables.css         # Design tokens and CSS variables
│   ├── _utilities.css         # All utility classes in one file
│   └── _functions.css         # Special functions and calculations
├── components/                # Component-specific styles
│   ├── _word-details.css      # Component styles 
│   └── _tabs.css              # Component styles
└── base/                      # Base styles
    └── _typography.css        # Typography rules
```

## How to Use

1. For general styling, use the utility classes:
   ```html
   <div class="flex justify-between items-center p-4">
     <h2 class="text-xl font-bold text-primary">Title</h2>
   </div>
   ```

2. For component-specific styles that can't be achieved with utilities:
   - Create a new file in the components directory
   - Import it in the styles/index.css file

3. To add a new design token:
   - Add it to utilities/_variables.css

4. For responsive design:
   - Use the mobile-first approach with breakpoint prefixes:
   ```html
   <div class="w-full md:w-1/2 lg:w-1/3">
     <!-- Full width on mobile, half on tablet, third on desktop -->
   </div>
   ```

## Documentation

For complete documentation, see [CSS-ARCHITECTURE.md](./CSS-ARCHITECTURE.md). 