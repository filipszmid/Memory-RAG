# --- General ---
.DEFAULT_GOAL := help

# COLORS
_BOLD    := \033[1m
_RED     := \033[31m
_GREEN   := \033[32m
_YELLOW  := \033[33m
_CYAN    := \033[36m
_DEFAULT := \033[0m

.PHONY: help
help: ## Show this help message
	@echo "$(_BOLD)Personal Memory Module - Available Commands:$(_DEFAULT)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(_CYAN)%-20s$(_DEFAULT) %s\n", $$1, $$2}'

# --- Local Development ---

.PHONY: dev-up
dev-up: ## Start the full stack with Docker Compose
	docker compose up --build -d
	@echo "$(_GREEN)Stack is starting at http://localhost:8501$(_DEFAULT)"

.PHONY: dev-down
dev-down: ## Stop the stack
	docker compose down

.PHONY: dev-logs
dev-logs: ## Follow logs for all services
	docker compose logs -f

# --- Processing ---

.PHONY: extract
extract: ## Run the bulk extraction CLI (Task 1)
	poetry run task-1-extract-facts --input ./example_conversations

.PHONY: test
test: ## Run the full test suite
	poetry run pytest tests/ -v

# --- Kubernetes ---

.PHONY: k8s-deploy
k8s-deploy: ## Deploy the stack to Kubernetes using Kustomize
	kubectl apply -k .

.PHONY: k8s-down
k8s-down: ## Stop all services in Kubernetes
	kubectl delete -k . || true

.PHONY: k8s-fix-images
k8s-fix-images: ## Rebuild images directly inside Minikube (Robust)
	@echo "Building Unified images inside Minikube..."
	@eval $$(minikube docker-env) && \
	docker build -t memory-agent-api:latest . && \
	docker build -t memory-agent-worker:latest . && \
	docker build -t memory-agent-ui:latest .

.PHONY: k8s-bounce-ui
k8s-bounce-ui: ## Restart specifically the UI pod (useful for CSS tweaks)
	kubectl delete pod -n memory-system -l app=ui
	@echo "$(_GREEN)Done! Wait for pods to restart.$(_DEFAULT)"

.PHONY: k8s-dashboard
k8s-dashboard: ## Open the Minikube dashboard to see the cluster
	minikube dashboard

.PHONY: k8s-rebuild-all
k8s-rebuild-all: ## [SUPER] Down, Build, Deploy, Wait, and Reset in one command
	make k8s-down
	@echo "Waiting for namespace to fully delete..."
	sleep 15
	make k8s-fix-images
	make k8s-deploy
	@echo "Waiting for pods to be ready..."
	kubectl wait --for=condition=ready pod -l app=api -n memory-system --timeout=120s
	kubectl wait --for=condition=ready pod -l app=ui -n memory-system --timeout=120s
	@echo "Waiting 5s for Ingress to settle..."
	sleep 5
	make k8s-reset
	@MINIKUBE_IP=$$(minikube ip); \
	echo "$(_GREEN)Full stack is UP on http://$$MINIKUBE_IP/$(_DEFAULT)"

.PHONY: k8s-reset
k8s-reset: ## Reset/Update the ElasticSearch index mapping in Kubernetes
	@MINIKUBE_IP=$$(minikube ip); \
	curl -X POST http://$$MINIKUBE_IP/api/reset-index

.PHONY: k8s-status
k8s-status: ## Check status of Kubernetes pods in memory-system
	kubectl get pods -n memory-system

.PHONY: k8s-watch
k8s-watch: ## Live watch of Kubernetes pods
	watch kubectl get pods -n memory-system

.PHONY: k8s-hosts
k8s-hosts: ## Print the lines to add to /etc/hosts for local routing
	@echo "$(_BOLD)Add these lines to your /etc/hosts file:$(_DEFAULT)"
	@MINIKUBE_IP=$$(minikube ip); \
	echo "$$MINIKUBE_IP    litellm.local"

.PHONY: k8s-tunnel
k8s-tunnel: ## Helper to start Minikube tunnel (requires sudo)
	@echo "$(_YELLOW)Running minikube tunnel... (Keep this terminal open)$(_DEFAULT)"
	minikube tunnel

# --- Maintenance ---

.PHONY: k8s-dash-it
k8s-dash-it: ## EMERGENCY: Direct tunnel to LiteLLM Dashboard (localhost:4000)
	@echo "$(_YELLOW)Opening direct tunnel to LiteLLM...$(_DEFAULT)"
	@echo "$(_GREEN)Open: http://localhost:4000/$(_DEFAULT)"
	kubectl port-forward -n memory-system svc/litellm-service 4000:4000

.PHONY: clean
clean: ## Clean up cache and temp files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf outputs/*
