// This file is for any additional JavaScript needed across the application
document.addEventListener('DOMContentLoaded', function() {
    // Add any global JavaScript functionality here
    
    // Example: Close flash messages
    document.querySelectorAll('.flash-close').forEach(button => {
        button.addEventListener('click', function() {
            this.parentElement.remove();
        });
    });
});