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
                    <p>Video downloaded successfully!</p>
                    <video controls>
                        <source src="${videoUrl}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                `);
            }
        });
    });
});