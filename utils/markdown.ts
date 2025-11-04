export const parseMarkdown = (text: string): string => {
  if (!text) return '';

  let html = text
    // Headers
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    // Bold
    .replace(/\*\*(.*?)\*\*/gim, '<strong>$1</strong>')
    // Italic
    .replace(/\*(.*?)\*/gim, '<em>$1</em>')
    // Unordered lists
    .replace(/^\s*\n\* (.*)/gim, '<ul>\n<li>$1</li>')
    .replace(/^\* (.*)/gim, '<li>$1</li>')
    // Ordered lists
    .replace(/^\s*\n\d\. (.*)/gim, '<ol>\n<li>$1</li>')
    .replace(/^\d\. (.*)/gim, '<li>$1</li>')
    // Close lists
    .replace(/<\/li>\n(?!<li>)/gim, '</li>\n</ul>')
    .replace(/<\/li>\n(?!<li>)/gim, '</li>\n</ol>')
    // Paragraphs (simple version)
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');

  // Cleanup for any accidentally opened but not closed list tags
  const openUl = (html.match(/<ul>/g) || []).length;
  const closeUl = (html.match(/<\/ul>/g) || []).length;
  if (openUl > closeUl) {
    html += '</ul>'.repeat(openUl - closeUl);
  }

  const openOl = (html.match(/<ol>/g) || []).length;
  const closeOl = (html.match(/<\/ol>/g) || []).length;
  if (openOl > closeOl) {
    html += '</ol>'.repeat(openOl - closeOl);
  }
  
  return html;
};
