# SRE Model Fine-tuning Makefile

.PHONY: install train test clean help

# Default model to use
MODEL ?= mlx-community/Qwen2.5-1.5B-Instruct-4bit
MODEL_PATH ?= ./sre-model

help: ## Show this help message
	@echo "SRE Model Fine-tuning with MLX"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	uv pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

train: ## Train the SRE model
	@echo "🚀 Training SRE model..."
	@echo "Model: $(MODEL)"
	@echo "Output: $(MODEL_PATH)"
	uv run python train_sre_model.py --train --model $(MODEL) --model_path $(MODEL_PATH)
	@echo "✅ Training completed!"

test: ## Test the trained model
	@echo "🧪 Testing SRE model..."
	uv run python train_sre_model.py --test --model $(MODEL) --model_path $(MODEL_PATH)
	@echo "✅ Testing completed!"

train-and-test: train test ## Train and then test the model

train-and-export: train export-ollama ## Train and export to Ollama

train-and-create: train create-ollama ## Train and create model in Ollama

quick-train: ## Quick training with fewer iterations (for testing)
	@echo "⚡ Quick training SRE model..."
	uv run python train_sre_model.py --train --model $(MODEL) --model_path $(MODEL_PATH)

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	rm -rf sre-model/
	rm -rf sre-model-ollama/
	rm -rf sre_training_data/
	rm -f sre_training_data.txt
	rm -f *.mlx
	rm -f OLLAMA_INSTRUCTIONS.txt
	@echo "✅ Cleanup completed!"

dataset-info: ## Show dataset information
	@echo "📊 SRE Dataset Information"
	@echo "=========================="
	@uv run python -c "import json; data=json.load(open('sre_dataset.json')); print(f'Total examples: {len(data)}'); print(f'First example: {data[0][\"instruction\"][:50]}...')"

# Advanced training options
train-custom: ## Train with custom parameters (set MODEL, MODEL_PATH, ITERS, LR)
	@echo "🚀 Custom training SRE model..."
	@echo "Model: $(MODEL)"
	@echo "Output: $(MODEL_PATH)"
	@echo "Iterations: $(ITERS)"
	@echo "Learning Rate: $(LR)"
	uv run python train_sre_model.py --train --model $(MODEL) --model_path $(MODEL_PATH)

export-ollama: ## Export fine-tuned model to Ollama format
	@echo "📦 Exporting SRE model to Ollama..."
	uv run python export_ollama.py --model $(MODEL) --adapter-path $(MODEL_PATH) --output ./sre-model-ollama
	@echo "✅ Export completed! Check OLLAMA_INSTRUCTIONS.txt for next steps."

create-ollama: export-ollama ## Export and create model in local Ollama
	@echo "🤖 Creating SRE model in Ollama..."
	@echo "🗑️  Removing existing model if it exists..."
	-ollama rm bruno-sre 2>/dev/null || true
	@echo "🔄 Creating new model..."
	ollama create bruno-sre -f ./sre-model-ollama/Modelfile
	@echo "✅ Model created in Ollama as 'bruno-sre'"
	@echo "🧪 Testing the model..."
	ollama run bruno-sre "Who is Bruno Lucena?"
	@echo "✅ Model is ready! Use: ollama run bruno-sre"
