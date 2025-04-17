# CSS Architecture Documentation

This project uses a robust CSS architecture built around utility classes and design tokens, following a modular approach for maintainability and scalability.

## Directory Structure

```
styles/
├── abstracts/                 # Legacy abstract styles
├── base/                      # Base styles
│   └── _typography.css        # Typography rules
├── components/                # Component-specific styles
│   ├── _tabs.css              # Component styles
│   └── _word-details.css      # Component styles
├── utilities/                 # Utility classes and functions
│   ├── _reset.css             # CSS reset/normalization
│   ├── _variables.css         # CSS variables and design tokens
│   ├── _functions.css         # Reusable functions
│   ├── _mixins.css            # Utility classes
│   └── _helpers.css           # Helper/utility classes
├── animations.css             # Animation keyframes and classes
├── layout.css                 # Layout and grid systems
├── mui-overrides.css          # Material-UI component overrides
└── index.css                  # Main stylesheet that imports all others
```

## Usage Guide

### 1. CSS Variables (Design Tokens)

Use CSS variables for consistent design throughout the application:

```css
/* Example usage */
.my-element {
  color: var(--color-primary);
  padding: var(--spacing-4);
  border-radius: var(--border-radius-md);
}
```

### 2. Utility Classes

The utility classes provide quick, reusable styling:

```html
<!-- Layout examples -->
<div class="container">
  <div class="flex justify-between items-center gap-md">
    <div class="w-full">Left content</div>
    <div class="w-full">Right content</div>
  </div>
</div>

<!-- Typography examples -->
<h2 class="text-xl font-bold text-primary">Section Title</h2>
<p class="text-sm text-secondary">Description text here</p>

<!-- Responsive examples -->
<div class="hidden md:block">Only visible on medium screens and up</div>
```

### 3. Aspect Ratio Utilities

For maintaining specific aspect ratios:

```html
<!-- Modern browsers -->
<div class="aspect-video">
  <img src="image.jpg" alt="Description" />
</div>

<!-- Legacy support -->
<div class="aspect-ratio-box aspect-ratio-16-9">
  <div class="aspect-ratio-box-inside">
    <img src="image.jpg" alt="Description" />
  </div>
</div>
```

### 4. Color with Opacity

For using colors with opacity:

```css
.element-with-opacity {
  background-color: rgba(var(--color-primary-rgb), 0.5);
}
```

## Best Practices

1. **Use Variables Over Hardcoded Values**
   ```css
   /* Good */
   margin: var(--spacing-4);
   
   /* Avoid */
   margin: 16px;
   ```

2. **Prefer Utility Classes for Common Patterns**
   ```html
   <!-- Good -->
   <div class="flex justify-between items-center p-4 rounded">...</div>
   
   <!-- Avoid creating custom classes for these common patterns -->
   <div class="custom-header">...</div>
   ```

3. **Component-Specific Styles**
   - Use component-specific styles only for unique styling needs
   - Keep component styles in the `styles/components/` directory

4. **Responsive Design**
   - Use the responsive utility classes with breakpoint prefixes
   - Example: `md:flex lg:grid`

5. **Dark Mode Support**
   - We use a `.dark` class applied to the document
   - Variables in `:root` have dark mode alternatives

## Making Changes

When modifying the CSS architecture:

1. Create backups before making significant changes:
   ```
   node src/backup-css.js
   ```

2. If you're adding new variables, add them to `_variables.css`

3. If you're adding utility classes, consider where they belong:
   - Layout utilities → `_mixins.css`
   - Color/design token utilities → `_variables.css`
   - Function-like utilities → `_functions.css`
   - General helpers → `_helpers.css`

4. When creating new components, leverage the utility classes first, then add component-specific styles only as needed. 