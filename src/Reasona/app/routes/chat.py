from flask import Blueprint, request, jsonify
from Reasona.services.reasona_service import run_reasona

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True)
    query = payload.get("query", "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    result = run_reasona(query)

    return jsonify(result)
