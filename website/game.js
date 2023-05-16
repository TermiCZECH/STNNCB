// help.js

// Function to toggle the help section
function toggleHelp() {
  const helpSection = document.querySelector('.help-section');
  helpSection.classList.toggle('hidden');
}

// Function to close the help section
function closeHelp() {
  const helpSection = document.querySelector('.help-section');
  helpSection.classList.add('hidden');
}

// Function to load explanations from a file
function loadExplanations() {
  fetch('explanations.json')
    .then(response => response.json())
    .then(data => {
      const helpContent = document.querySelector('.help-content');
      helpContent.innerHTML = ''; // Clear the existing content

      // Loop through the explanations data and create HTML elements for each explanation
      for (const key in data) {
        if (data.hasOwnProperty(key)) {
          const explanation = data[key];
          const explanationElement = document.createElement('div');
          explanationElement.innerHTML = `<h3>${key}</h3><p>${explanation}</p>`;
          helpContent.appendChild(explanationElement);
        }
      }
    })
    .catch(error => {
      console.error('Error:', error);
    });
}

// Call the function to load the explanations on page load
window.addEventListener('DOMContentLoaded', loadExplanations);