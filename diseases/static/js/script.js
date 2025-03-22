document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("prediction-form");
  const results = document.getElementById("results");
  const predictionText = document.getElementById("prediction-text");
  const probabilityText = document.getElementById("probability-text");
  const probabilityBar = document.getElementById("probability-bar");
  const resultBox = document.getElementById("result-box");
  const shapPlot = document.getElementById("shap-plot"); // Element for SHAP plot
  const forcePlot = document.getElementById("force-plot"); // Element for feature importance plot

  // Sample data for testing
  const sampleData = {
    Pregnancies: 6,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 0,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50,
  };

  // Add sample data button
  const formContainer = document.querySelector(".form-container");
  const sampleButton = document.createElement("button");
  sampleButton.textContent = "Use Sample Data";
  sampleButton.className = "submit-btn";
  sampleButton.style.marginTop = "20px";
  sampleButton.style.backgroundColor = "#2ecc71";

  sampleButton.addEventListener("click", function (e) {
    e.preventDefault();

    // Fill the form with sample data
    for (const [key, value] of Object.entries(sampleData)) {
      document.getElementById(key).value = value;
    }
  });

  formContainer.appendChild(sampleButton);

  // Form submission
  form.addEventListener("submit", function (e) {
    e.preventDefault();

    // Show loading state
    const submitBtn = form.querySelector(".submit-btn");
    const originalText = submitBtn.textContent;
    submitBtn.textContent = "Predicting...";
    submitBtn.disabled = true;

    // Get form data
    const formData = new FormData(form);

    // Send prediction request to the /predict endpoint

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        // Update UI with prediction results and plots

        const prediction = data.prediction;
        const probability = data.probability;

        // Update prediction text
        predictionText.textContent =
          prediction === 1
            ? "Positive (High Risk of Diabetes)"
            : "Negative (Low Risk of Diabetes)";
        resultBox.className =
          prediction === 1 ? "result-box positive" : "result-box negative";

        // Update probability
        probabilityText.textContent = `Probability: ${probability.toFixed(2)}%`;
        probabilityBar.style.width = `${probability}%`;

        // Update plots
        shapPlot.src = `data:image/png;base64,${data.shap_plot}`;
        forcePlot.src = `data:image/png;base64,${data.force_plot}`;

        // Show results
        results.style.display = "block";

        // Scroll to results
        results.scrollIntoView({ behavior: "smooth" });

        // Reset button
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      })
      .catch((error) => {
        console.error("Error:", error);
        alert("An error occurred during prediction. Please try again.");

        // Reset button
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
      });
  });
});
