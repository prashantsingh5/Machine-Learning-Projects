$(document).ready(function() {
    Dropzone.options.dropzone = {
        acceptedFiles: ".jpeg,.jpg,.png,.gif",
        maxFiles: 1,
        init: function() {
            this.on("success", function(file, response) {
                if (response.error) {
                    $('#error').show();
                } else {
                    $('#resultHolder').append("<h4>Prediction: " + response.prediction + "</h4>");
                    // Process and display the probabilities
                    // You can create your logic to display the probabilities
                }
            });
        }
    };

    $('#submitBtn').click(function() {
        $('#dropzone').get(0).dropzone.processQueue();  // Manually trigger file upload
    });
});
