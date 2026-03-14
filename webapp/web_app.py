import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from app_p1_web import PDF_FOLDER, build_chatbot, ask_question

app = Flask(__name__)
app.config["SECRET_KEY"] = "pdf-chatbot-secret-key"
app.config["UPLOAD_FOLDER"] = PDF_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}

chat_chain = None
current_mode = "open_source"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdfs():
    global chat_chain, current_mode

    mode = request.form.get("mode", "open_source").strip().lower()
    if mode not in ("openai", "open_source"):
        return jsonify({"success": False, "error": "Invalid mode."}), 400

    if "pdfs" not in request.files:
        return jsonify({"success": False, "error": "No files uploaded."}), 400

    files = request.files.getlist("pdfs")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"success": False, "error": "No selected files."}), 400

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    saved_count = 0
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            saved_count += 1

    if saved_count == 0:
        return jsonify({"success": False, "error": "Only PDF files are allowed."}), 400

    try:
        chat_chain, chunk_count, pdf_count = build_chatbot(mode)
        current_mode = mode
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    return jsonify(
        {
            "success": True,
            "message": f"Uploaded {saved_count} PDF(s). Chatbot is ready.",
            "pdf_count": pdf_count,
            "chunk_count": chunk_count,
            "mode": current_mode,
        }
    )


@app.route("/ask", methods=["POST"])
def ask():
    global chat_chain

    if chat_chain is None:
        return jsonify(
            {
                "success": False,
                "error": "Please upload and process PDFs first."
            }
        ), 400

    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    show_sources = bool(data.get("show_sources", False))

    if not question:
        return jsonify({"success": False, "error": "Question is empty."}), 400

    try:
        result = ask_question(chat_chain, question, show_sources=show_sources)
        return jsonify(
            {
                "success": True,
                "answer": result["answer"],
                "sources": result["sources"],
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs(PDF_FOLDER, exist_ok=True)
    app.run(debug=True)