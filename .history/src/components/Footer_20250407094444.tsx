import React from 'react';
import './Footer.css';

interface FooterProps {
  extraLinks?: { label: string; url: string }[];
}

const Footer: React.FC<FooterProps> = ({ extraLinks }) => {
  const currentYear = new Date().getFullYear();
  
  return (
    <footer className="footer">
      <div className="footer-content">
        <p className="copyright">
          © {currentYear} Filipino Root Word Explorer. All Rights Reserved.
        </p>
        
        {extraLinks && extraLinks.length > 0 && (
          <div className="footer-links">
            {extraLinks.map((link, index) => (
              <React.Fragment key={link.url}>
                <a 
                  href={link.url} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="footer-link"
                >
                  {link.label}
                </a>
                {index < extraLinks.length - 1 && <span className="divider">•</span>}
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    </footer>
  );
};

export default Footer; 