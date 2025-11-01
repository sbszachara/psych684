function showPDF(pdfFile) {
      const container = document.getElementById('pdf-display');
      container.innerHTML = `
        <iframe src="${pdfFile}" allowfullscreen></iframe>
        <p style="margin-top: 10px;">
          Your browser may not support embedded PDFs. 
          <a href="${pdfFile}" target="_blank" rel="noopener">Download the PDF</a> instead.
        </p>
      `;
    }



window.onload = function() {
  showPDF('psyc684_abstract.pdf');
};