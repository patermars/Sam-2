$(document).ready(function(){
    $('#videoForm').on('submit', function(e){
        e.preventDefault();
        var videoUrl = $('#videoUrl').val();
        $('#videoContainer').html(`<p>Downloading video: ${videoUrl}</p>`);
        
        $.post('/download', { videoUrl: videoUrl }, function(response){
            if (response.error) {
                $('#videoContainer').html(`<p>Error: ${response.error}</p>`);
            } else {
                var videoUrl = `/video/${response.filename}`;
                $('#videoContainer').html(`
                    <div class="video-container">
                        <video id="videoElement" controls>
                            <source src="${videoUrl}" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                `);

                $('#videoElement').on('click', function(event){
                    var videoElement = this;
                    var rect = videoElement.getBoundingClientRect();

                    var x = event.clientX - rect.left; // X-coordinate relative to the video element
                    var y = event.clientY - rect.top;  // Y-coordinate relative to the video element

                    var currentTime = videoElement.currentTime;

                    // Send coordinates and current time to Flask server
                    $.ajax({
                        url: '/coordinates',
                        type: 'POST',
                        contentType: 'application/json',
                        data: JSON.stringify({ x: x, y: y, currentTime: currentTime }),
                        success: function(response) {
                            console.log('Coordinates sent successfully:', response);
                        }
                    });
                });
            }
        });
    });
});