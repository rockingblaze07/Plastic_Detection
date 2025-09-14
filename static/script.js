document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("upload-form");
    const output = document.getElementById("output");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById("file-upload");
        if (fileInput.files.length === 0) {
            output.innerHTML = "<p>Please select an image first.</p>";
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                output.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
            } else {
                output.innerHTML = `
                    <p><strong>Prediction:</strong> ${data.label}</p>
                    <p><strong>Confidence:</strong> ${data.confidence}%</p>
                `;
            }
        } catch (err) {
            output.innerHTML = `<p style="color:red;">Something went wrong.</p>`;
        }
    });
});
