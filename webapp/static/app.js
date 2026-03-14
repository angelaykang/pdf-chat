const uploadBtn = document.getElementById("uploadBtn");
const sendBtn = document.getElementById("sendBtn");
const pdfInput = document.getElementById("pdfInput");
const questionInput = document.getElementById("questionInput");
const chatWindow = document.getElementById("chatWindow");
const statusText = document.getElementById("statusText");
const modeSelect = document.getElementById("mode");
const showSourcesCheckbox = document.getElementById("showSources");

function addMessage(text, sender, sources = []) {
    const wrapper = document.createElement("div");
    wrapper.className = `message ${sender}`;

    const card = document.createElement("div");
    card.className = "message-card";

    const role = document.createElement("div");
    role.className = "message-role";
    role.textContent = sender === "user" ? "You" : "Assistant";

    const body = document.createElement("div");
    body.className = "message-text";
    body.textContent = text;

    card.appendChild(role);
    card.appendChild(body);

    if (sender === "bot" && sources.length > 0) {
        const sourceBox = document.createElement("div");
        sourceBox.className = "sources";

        let sourceText = "Source snippets\n";
        sources.forEach((src) => {
            sourceText += `\n[${src.index}] ${src.snippet}...`;
        });

        sourceBox.textContent = sourceText;
        card.appendChild(sourceBox);
    }

    wrapper.appendChild(card);
    chatWindow.appendChild(wrapper);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function uploadPDFs() {
    const files = pdfInput.files;
    if (!files || files.length === 0) {
        alert("Please select at least one PDF file.");
        return;
    }

    const formData = new FormData();
    for (const file of files) {
        formData.append("pdfs", file);
    }
    formData.append("mode", modeSelect.value);

    statusText.textContent = "Uploading and processing PDFs...";
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Processing...";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!data.success) {
            statusText.textContent = `Error: ${data.error}`;
            addMessage(`Upload failed: ${data.error}`, "bot");
            return;
        }

        statusText.textContent =
            `${data.message} Loaded ${data.pdf_count} PDFs and ${data.chunk_count} chunks.`;

        addMessage(
            "Your PDFs are ready. Ask a question whenever you are ready.",
            "bot"
        );
    } catch (error) {
        statusText.textContent = `Upload failed: ${error.message}`;
        addMessage(`Upload failed: ${error.message}`, "bot");
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload and Process";
    }
}

async function sendQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    addMessage(question, "user");
    questionInput.value = "";
    sendBtn.disabled = true;
    sendBtn.textContent = "Sending...";

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question: question,
                show_sources: showSourcesCheckbox.checked,
            }),
        });

        const data = await response.json();

        if (!data.success) {
            addMessage(`Error: ${data.error}`, "bot");
            return;
        }

        addMessage(data.answer, "bot", data.sources || []);
    } catch (error) {
        addMessage(`Request failed: ${error.message}`, "bot");
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "Send";
        questionInput.focus();
    }
}

uploadBtn.addEventListener("click", uploadPDFs);
sendBtn.addEventListener("click", sendQuestion);

questionInput.addEventListener("keydown", function (event) {
    if (event.key === "Enter") {
        sendQuestion();
    }
});