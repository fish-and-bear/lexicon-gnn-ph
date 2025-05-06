# CSS Architecture Documentation

This project uses a simplified, modular CSS architecture built around utility-first design with CSS custom properties (variables).

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

## Key Principles

1. **Single Source of Truth**: Each style has one clear location
2. **Design Token System**: All values come from CSS variables
3. **Utility-First**: Use utility classes before custom CSS
4. **Minimal Specificity**: Keep selectors simple to avoid conflicts
5. **Self-Documenting**: Class names describe their function

## Usage Guide

### 1. Design Tokens (CSS Variables)

All design values come from variables defined in `_variables.css`:

```css
.my-element {
  color: var(--color-primary);
  padding: var(--spacing-4);
  border-radius: var(--border-radius-md);
}
```

### 2. Utility Classes

Use utility classes for common style patterns:

```html
<!-- Layout example -->
<div class="container mx-auto p-4">
  <div class="flex justify-between items-center gap-md">
    <div class="w-full md:w-1/2">Left content</div>
    <div class="w-full md:w-1/2">Right content</div>
  </div>
</div>

<!-- Typography example -->
<h2 class="text-xl font-bold text-primary mb-4">Title</h2>
<p class="text-secondary mb-2">Description</p>
```

### 3. Component Styles

Only create component-specific styles when utility classes don't suffice:

```css
/* In components/_custom-widget.css */
.custom-widget {
  /* Unique styles that can't be expressed with utilities */
}
```

### 4. Special Functions

For advanced patterns like aspect ratios or color opacity:

```css
/* Aspect ratio */
.product-image {
  aspect-ratio: 1/1; /* Modern browsers */
}

/* Legacy support */
.product-image-legacy {
  composes: aspect-ratio-box aspect-ratio-1-1;
}

/* Colors with opacity */
.overlay {
  background-color: rgba(var(--color-primary-rgb), 0.5);
}
```

## Responsive Design

We use a mobile-first approach with breakpoint prefixes:

```html
<div class="hidden md:block lg:flex">
  <!-- Hidden on mobile, block on tablet, flex on desktop -->
</div>
```

## Best Practices

1. **Use the cascade appropriately**
   - Keep inheritance for meaningful relationships
   - Avoid deep nesting of selectors

2. **Add new styles in the right place**
   - New design tokens → _variables.css
   - New utility classes → _utilities.css
   - Component-specific styles → components/
   
3. **Keep specificity low**
   - Avoid ID selectors (#id)
   - Minimize use of !important
   - Use classes over attributes when possible

4. **Make the easy changes first**
   - Consider using existing utilities before creating new ones
   - Compose complex components from simple utility classes 