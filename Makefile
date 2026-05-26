SHELL := /bin/bash
SHELLFLAGS := -c
PYTHON ?= python3
MODEL ?= gpt-4o-mini
RESULTS_DIR ?= results
SLEEP = 1
RETRY = 3
WELDPROMPT_KS ?= 3 5 7 9
WELDPROMPT_STRATEGIES ?= similarity diverse balanced balanced-diverse
VARIANT_FLAGS ?=
ASSIGNMENT_VARIANT_REPORTS = \
	$(RESULTS_DIR)/weldprompt-similarity-k3-class-report.xlsx \
	$(RESULTS_DIR)/weldprompt-similarity-k7-class-report.xlsx \
	$(RESULTS_DIR)/weldprompt-similarity-k9-class-report.xlsx \
	$(RESULTS_DIR)/weldprompt-diverse-k5-class-report.xlsx \
	$(RESULTS_DIR)/weldprompt-balanced-k5-class-report.xlsx \
	$(RESULTS_DIR)/weldprompt-balanced-diverse-k5-class-report.xlsx

LORA_MODEL ?= $(MODEL)
LORA_DATA_DIR ?= finetune/data/llava-realworld
LORA_OUTPUT_DIR ?= finetune/outputs/llava-realworld-lora
LORA_RESULTS_DIR ?= results_lora
LORA_EPOCHS ?= 3
LORA_BATCH_SIZE ?= 1
LORA_GRAD_ACCUM ?= 8
LORA_LR ?= 2e-4
LORA_MAX_LENGTH ?= 1536
LORA_NUM_RUNS ?= 3
LORA_MAX_NEW_TOKENS ?= 64
LORA_LIMIT ?=
LORA_LIMIT_FLAG = $(if $(LORA_LIMIT),--limit $(LORA_LIMIT),)
LORA_PREP_FLAGS ?=




default: all

GUIDS = $(shell $(PYTHON) -c "import json;print(' '.join(sorted({x['guid'] for x in json.load(open('data/guids.json'))})))")
WELDPROMPT_VARIANT_REPORTS = $(foreach strategy,$(WELDPROMPT_STRATEGIES),$(foreach k,$(WELDPROMPT_KS),$(RESULTS_DIR)/weldprompt-$(strategy)-k$(k)-class-report.xlsx))

########################
# MAIN PIPELINE
######################## 

all: zero_shot_responses \
	embeddings \
	zero_shot_dist_report zero_shot_class_report \
	image_embeddings cot_precompute \
	medprompt_responses \
	medprompt_dist_report medprompt_class_report


########################
# IMAGE EMBEDDINGS
########################

IMAGE_EMBEDDINGS = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/image-embeddings/$(guid).json)

image_embeddings: $(IMAGE_EMBEDDINGS)

$(RESULTS_DIR)/image-embeddings/%.json: data/pics/%.png
	@mkdir -p $(RESULTS_DIR)/image-embeddings
	@if [ -f $@ ]; then \
		echo "✓ Skipping image embedding $*"; \
	else \
		echo "Running image embedding $*"; \
		$(PYTHON) scripts/image_embeddings.py --guid $* --out $@; \
		sleep $(SLEEP); \
	fi


########################
# ZERO SHOT RESPONSES
########################

RESPONSE_FILES = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/zero-shot-responses/$(guid).json)

zero_shot_responses: $(RESPONSE_FILES)

$(RESULTS_DIR)/zero-shot-responses/%.json: data/data/%.json data/pics/%.png
	@mkdir -p $(RESULTS_DIR)/zero-shot-responses
	@if [ -f $@ ]; then \
		echo "✓ Skipping zero-shot $*"; \
	else \
		for i in $$(seq 1 $(RETRY)); do \
			echo "Running zero-shot $* (try $$i)"; \
			$(PYTHON) scripts/generate_response.py --guid $* --model ${MODEL} --out $@ && break; \
			echo "Retry $$i failed"; \
			sleep 5; \
		done; \
		sleep $(SLEEP); \
	fi


########################
# COT PRECOMPUTE
########################

cot_precompute: $(RESULTS_DIR)/cot_data.json

$(RESULTS_DIR)/cot_data.json: $(IMAGE_EMBEDDINGS) $(RESPONSE_FILES)
	@mkdir -p $(RESULTS_DIR)
	@if [ -f $@ ]; then \
		echo "✓ Skipping cot precompute"; \
	else \
		echo "Running cot precompute"; \
		RESULTS_ROOT=$(RESULTS_DIR) $(PYTHON) scripts/precompute_cots.py --guids data/guids.json --out $@; \
	fi


########################
# MEDPROMPT RESPONSES
########################

MEDPROMPT_FILES = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/medprompt-responses/$(guid).json)

medprompt_responses: $(MEDPROMPT_FILES)

$(RESULTS_DIR)/medprompt-responses/%.json: data/data/%.json data/pics/%.png data/guids.json $(RESULTS_DIR)/cot_data.json
	@mkdir -p $(RESULTS_DIR)/medprompt-responses
	@if [ -f $@ ]; then \
		echo "✓ Skipping medprompt $*"; \
	else \
		for i in $$(seq 1 $(RETRY)); do \
			echo "Running medprompt $* (try $$i)"; \
			RESULTS_ROOT=$(RESULTS_DIR) $(PYTHON) scripts/generate_medprompt_response.py \
			--guid $* \
			--model ${MODEL} \
			--out $@ \
			--guids data/guids.json \
			--cotdata $(RESULTS_DIR)/cot_data.json && break; \
			echo "Retry $$i failed"; \
			sleep 5; \
		done; \
		sleep $(SLEEP); \
	fi


########################
# EMBEDDINGS
########################

ZERO_SHOT_RESPONSE_EMBEDDING_FILES = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/zero-shot-responses-embeddings/$(guid).json)

zero_shot_response_embeddings: $(ZERO_SHOT_RESPONSE_EMBEDDING_FILES)

$(RESULTS_DIR)/zero-shot-responses-embeddings/%.json: $(RESULTS_DIR)/zero-shot-responses/%.json
	@mkdir -p $(RESULTS_DIR)/zero-shot-responses-embeddings
	@if [ -f $@ ]; then \
		echo "✓ Skipping zero-shot embedding $*"; \
	else \
		echo "Running zero-shot embedding $*"; \
		$(PYTHON) scripts/create_embeddings.py --input $< --out $@; \
		sleep $(SLEEP); \
	fi


MEDPROMPT_RESPONSE_EMBEDDING_FILES = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/medprompt-responses-embeddings/$(guid).json)

medprompt_response_embeddings: $(MEDPROMPT_RESPONSE_EMBEDDING_FILES)

$(RESULTS_DIR)/medprompt-responses-embeddings/%.json: $(RESULTS_DIR)/medprompt-responses/%.json
	@mkdir -p $(RESULTS_DIR)/medprompt-responses-embeddings
	@if [ -f $@ ]; then \
		echo "✓ Skipping medprompt embedding $*"; \
	else \
		echo "Running medprompt embedding $*"; \
		$(PYTHON) scripts/create_embeddings.py --input $< --out $@; \
		sleep $(SLEEP); \
	fi


DATA_EMBEDDING_FILES = $(foreach guid,$(GUIDS),$(RESULTS_DIR)/data-embeddings/$(guid).json)

data_embeddings: $(DATA_EMBEDDING_FILES)

$(RESULTS_DIR)/data-embeddings/%.json: data/data/%.json
	@mkdir -p $(RESULTS_DIR)/data-embeddings
	@if [ -f $@ ]; then \
		echo "✓ Skipping data embedding $*"; \
	else \
		echo "Running data embedding $*"; \
		$(PYTHON) scripts/create_embeddings.py --input $< --out $@; \
		sleep $(SLEEP); \
	fi


embeddings: zero_shot_response_embeddings medprompt_response_embeddings data_embeddings


########################
# REPORTS
########################

DATA_FILES = $(foreach guid,$(GUIDS),data/data/$(guid).json)

zero_shot_class_report: $(RESULTS_DIR)/zero-shot-class-report.xlsx

$(RESULTS_DIR)/zero-shot-class-report.xlsx: $(RESPONSE_FILES) $(DATA_FILES) data/guids.json
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/test_classification.py \
	--guids data/guids.json \
	--truth data/data \
	--pred $(RESULTS_DIR)/zero-shot-responses \
	-o $@


zero_shot_dist_report: $(RESULTS_DIR)/zero-shot-dist-report.xlsx

$(RESULTS_DIR)/zero-shot-dist-report.xlsx: $(DATA_EMBEDDING_FILES) $(ZERO_SHOT_RESPONSE_EMBEDDING_FILES) data/guids.json
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/test_distances.py \
	--guids data/guids.json \
	--truth $(RESULTS_DIR)/data-embeddings \
	--pred $(RESULTS_DIR)/zero-shot-responses-embeddings \
	--out $@


medprompt_class_report: $(RESULTS_DIR)/medprompt-class-report.xlsx

$(RESULTS_DIR)/medprompt-class-report.xlsx: $(MEDPROMPT_FILES) $(DATA_FILES) data/guids.json
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/test_classification.py \
	--guids data/guids.json \
	--truth data/data \
	--pred $(RESULTS_DIR)/medprompt-responses \
	-o $@


medprompt_dist_report: $(RESULTS_DIR)/medprompt-dist-report.xlsx

$(RESULTS_DIR)/medprompt-dist-report.xlsx: $(DATA_EMBEDDING_FILES) $(MEDPROMPT_RESPONSE_EMBEDDING_FILES) data/guids.json
	@mkdir -p $(RESULTS_DIR)
	$(PYTHON) scripts/test_distances.py \
	--guids data/guids.json \
	--truth $(RESULTS_DIR)/data-embeddings \
	--pred $(RESULTS_DIR)/medprompt-responses-embeddings \
	--out $@


########################
# DYNAMIC WELDPROMPT EXPERIMENTS
########################

.PHONY: llava_baseline_class weldprompt_variants weldprompt_summary llava_dynamic_experiments \
	weldprompt_assignment_variants weldprompt_assignment_summary llava_assignment_experiments check_prompt_echo \
	lora_prepare lora_train lora_eval_base lora_eval_adapter lora_reports lora_experiments \
	lora_smoke_train lora_smoke_eval

llava_baseline_class: zero_shot_responses image_embeddings cot_precompute medprompt_responses zero_shot_class_report medprompt_class_report

define RUN_WELDPROMPT_VARIANT
$(RESULTS_DIR)/weldprompt-$(1)-k$(2)-class-report.xlsx: $(RESULTS_DIR)/cot_data.json
	RESULTS_ROOT=$(RESULTS_DIR) $(PYTHON) scripts/run_weldprompt_variant.py \
	--model "$(MODEL)" \
	--cotdata $(RESULTS_DIR)/cot_data.json \
	--outdir $(RESULTS_DIR)/weldprompt-$(1)-k$(2) \
	--k $(2) \
	--selection-strategy $(1) \
	--sleep $(SLEEP) \
	--retries $(RETRY) \
	$(VARIANT_FLAGS)
endef

$(foreach strategy,$(WELDPROMPT_STRATEGIES),$(foreach k,$(WELDPROMPT_KS),$(eval $(call RUN_WELDPROMPT_VARIANT,$(strategy),$(k)))))

weldprompt_variants: $(WELDPROMPT_VARIANT_REPORTS)

weldprompt_summary: $(RESULTS_DIR)/weldprompt-summary.csv

$(RESULTS_DIR)/weldprompt-summary.csv: $(RESULTS_DIR)/zero-shot-class-report.xlsx $(RESULTS_DIR)/medprompt-class-report.xlsx $(WELDPROMPT_VARIANT_REPORTS)
	$(PYTHON) scripts/summarize_class_reports.py \
	--reports $^ \
	--out $@

llava_dynamic_experiments: llava_baseline_class weldprompt_variants weldprompt_summary

weldprompt_assignment_variants: $(ASSIGNMENT_VARIANT_REPORTS)

weldprompt_assignment_summary: $(RESULTS_DIR)/assignment-weldprompt-summary.csv

$(RESULTS_DIR)/assignment-weldprompt-summary.csv: $(RESULTS_DIR)/zero-shot-class-report.xlsx $(RESULTS_DIR)/medprompt-class-report.xlsx $(ASSIGNMENT_VARIANT_REPORTS)
	$(PYTHON) scripts/summarize_class_reports.py \
	--reports $^ \
	--out $@

llava_assignment_experiments: llava_baseline_class weldprompt_assignment_variants weldprompt_assignment_summary

check_prompt_echo:
	@matches=$$(grep -R "\[INST\]" -n $(RESULTS_DIR) --include="*.json" | head -20); \
	if [ -n "$$matches" ]; then \
		echo "$$matches"; \
		echo "Found prompt echo markers above."; \
		exit 1; \
	else \
		echo "No [INST] prompt echo markers found in $(RESULTS_DIR)."; \
	fi


########################
# LLAVA LORA FINETUNING
########################

lora_prepare:
	$(PYTHON) finetune/prepare_llava_lora_data.py \
	--outdir $(LORA_DATA_DIR) \
	$(LORA_PREP_FLAGS)

lora_train: lora_prepare
	$(PYTHON) finetune/train_llava_lora.py \
	--model "$(LORA_MODEL)" \
	--train-jsonl $(LORA_DATA_DIR)/train.jsonl \
	--eval-jsonl $(LORA_DATA_DIR)/dev.jsonl \
	--output-dir $(LORA_OUTPUT_DIR) \
	--epochs $(LORA_EPOCHS) \
	--batch-size $(LORA_BATCH_SIZE) \
	--grad-accum $(LORA_GRAD_ACCUM) \
	--lr $(LORA_LR) \
	--max-length $(LORA_MAX_LENGTH) \
	--gradient-checkpointing

lora_smoke_train: lora_prepare
	$(PYTHON) finetune/train_llava_lora.py \
	--model "$(LORA_MODEL)" \
	--train-jsonl $(LORA_DATA_DIR)/train.jsonl \
	--eval-jsonl $(LORA_DATA_DIR)/dev.jsonl \
	--output-dir $(LORA_OUTPUT_DIR)-smoke \
	--epochs 0.03 \
	--batch-size 1 \
	--grad-accum 1 \
	--max-length 1024 \
	--limit-train 6 \
	--limit-eval 3 \
	--gradient-checkpointing

lora_eval_base:
	$(PYTHON) finetune/evaluate_llava_lora.py \
	--model "$(LORA_MODEL)" \
	--guids data/guids.json \
	--outdir $(LORA_RESULTS_DIR)/zero-shot-direct \
	--num-runs $(LORA_NUM_RUNS) \
	--max-new-tokens $(LORA_MAX_NEW_TOKENS) \
	$(LORA_LIMIT_FLAG)

lora_eval_adapter:
	$(PYTHON) finetune/evaluate_llava_lora.py \
	--model "$(LORA_MODEL)" \
	--adapter-dir "$(LORA_OUTPUT_DIR)" \
	--guids data/guids.json \
	--outdir $(LORA_RESULTS_DIR)/lora-realworld \
	--num-runs $(LORA_NUM_RUNS) \
	--max-new-tokens $(LORA_MAX_NEW_TOKENS) \
	$(LORA_LIMIT_FLAG)

lora_smoke_eval:
	$(PYTHON) finetune/evaluate_llava_lora.py \
	--model "$(LORA_MODEL)" \
	--adapter-dir "$(LORA_OUTPUT_DIR)-smoke" \
	--guids data/guids.json \
	--outdir $(LORA_RESULTS_DIR)/lora-smoke \
	--limit 2 \
	--overwrite

lora_reports:
	@mkdir -p $(LORA_RESULTS_DIR)
	$(PYTHON) scripts/test_classification.py \
	--guids data/guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/zero-shot-direct \
	-o $(LORA_RESULTS_DIR)/zero-shot-direct-all-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids data/guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/lora-realworld \
	-o $(LORA_RESULTS_DIR)/lora-realworld-all-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/train-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/zero-shot-direct \
	-o $(LORA_RESULTS_DIR)/zero-shot-direct-train-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/train-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/lora-realworld \
	-o $(LORA_RESULTS_DIR)/lora-realworld-train-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/dev-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/zero-shot-direct \
	-o $(LORA_RESULTS_DIR)/zero-shot-direct-dev-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/dev-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/lora-realworld \
	-o $(LORA_RESULTS_DIR)/lora-realworld-dev-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/web-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/zero-shot-direct \
	-o $(LORA_RESULTS_DIR)/zero-shot-direct-web-class-report.xlsx
	$(PYTHON) scripts/test_classification.py \
	--guids $(LORA_DATA_DIR)/web-guids.json \
	--truth data/data \
	--pred $(LORA_RESULTS_DIR)/lora-realworld \
	-o $(LORA_RESULTS_DIR)/lora-realworld-web-class-report.xlsx
	$(PYTHON) scripts/summarize_class_reports.py \
	--reports \
	$(LORA_RESULTS_DIR)/zero-shot-direct-all-class-report.xlsx \
	$(LORA_RESULTS_DIR)/lora-realworld-all-class-report.xlsx \
	$(LORA_RESULTS_DIR)/zero-shot-direct-train-class-report.xlsx \
	$(LORA_RESULTS_DIR)/lora-realworld-train-class-report.xlsx \
	$(LORA_RESULTS_DIR)/zero-shot-direct-dev-class-report.xlsx \
	$(LORA_RESULTS_DIR)/lora-realworld-dev-class-report.xlsx \
	$(LORA_RESULTS_DIR)/zero-shot-direct-web-class-report.xlsx \
	$(LORA_RESULTS_DIR)/lora-realworld-web-class-report.xlsx \
	--out $(LORA_RESULTS_DIR)/lora-summary.csv

lora_experiments: lora_train lora_eval_base lora_eval_adapter lora_reports
