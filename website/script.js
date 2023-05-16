document.addEventListener("DOMContentLoaded", function() {
    const optionsButton = document.getElementById("optionsButton");
    const optionsDropdown = document.getElementById("optionsDropdown");

    // Fetch the list of options from a JSON file
    fetch("options.json")
        .then(response => response.json())
        .then(data => {
            // Populate the dropdown menu with the options
            data.forEach(option => {
                const optionElement = document.createElement("option");
                optionElement.value = option;
                optionElement.text = option;
                optionsDropdown.appendChild(optionElement);
            });
        });

    // Add an event listener to the button
    optionsButton.addEventListener("click", function() {
        const selectedOption = optionsDropdown.value;
        // Send the selected option to the server for updating the configuration
        updateConfig(selectedOption);
    });

    // Function to send the selected option to the server for updating the configuration
    function updateConfig(option) {
        fetch("/update_config", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                selectedOption: option
            })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Config updated:", data);
            // Display a success message or perform any additional actions
        })
        .catch(error => {
            console.error("Error updating config:", error);
            // Display an error message or handle the error gracefully
        });
    }
});  