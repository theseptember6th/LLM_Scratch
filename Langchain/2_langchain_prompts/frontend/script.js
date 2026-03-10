const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async function (event) {
  event.preventDefault();
  const data = {
    message: document.getElementById("user_input").value,
  };

  try {
    const response = await fetch("http://localhost:8000/summarize", {
      method: "POST",
      headers: {
        "content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (response.ok) {
      const result = await response.json();
      resultDiv.innerHTML = `
  <h3>Response from LLM</h3>
  <p><strong>Predicted Output:</strong> ${result.response.output}</p>
`;
    }
  } catch (error) {
    result.innerHTML = `got an ${error}`;
  }
});
