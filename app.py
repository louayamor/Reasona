from flask import Flask, request, jsonify, render_template
from Reasona.services.reasona_service import ReasonaService
from Reasona.pipeline.inference_pipeline import InferencePipeline
from Reasona.pipeline.reranking_pipeline import RerankingPipeline
from Reasona.pipeline.retrieval_pipeline import RetrievalPipeline
from Reasona.config.config_manager import ConfigurationManager
from Reasona.utils.logger import setup_logger
import traceback
import os

logger = setup_logger(
    "reasona_flask",
    "logs/service/reasona_flask.json"
)

app = Flask(__name__, static_folder="static", template_folder="templates")

cfg_manager = ConfigurationManager()
logger.info("ConfigurationManager initialized")

retrieval_pipeline = RetrievalPipeline(cfg_manager.get_retrieval_config())
reranking_pipeline = RerankingPipeline(retrieval_pipeline, cfg_manager)
inference_pipeline = InferencePipeline(reranking_pipeline, cfg_manager)
service = ReasonaService(inference_pipeline)
logger.info("ReasonaService initialized with InferencePipeline")

@app.route("/")
def index():
    """Serve the main HTML page"""
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    """Forward queries from frontend to ReasonaService and return JSON object"""
    try:
        data = request.get_json(force=True)
        query_text = data.get("query")

        if not query_text or not isinstance(query_text, str):
            return jsonify({"error": "Missing or invalid 'query' field"}), 400

        response = service.answer(query_text)

        answer = response.get("answer")
        if isinstance(answer, dict):
            answer = answer.get("text") or str(answer)
        elif not isinstance(answer, str):
            answer = str(answer)

        return jsonify({
            "query": query_text,
            "answer": answer,
            "metadata": response.get("metadata", {}) 
        })

    except Exception as e:
        logger.exception("Query endpoint error")
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500



@app.route("/health", methods=["GET"])
def health():
    """Basic health check"""
    return jsonify({"status": "healthy"}), 200


@app.route("/ready", methods=["GET"])
def ready():
    """Readiness check: test ReasonaService"""
    try:
        _ = service.answer("ping")
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        logger.exception("Readiness check failed")
        return jsonify({"status": "not ready", "error": str(e)}), 503

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
