# FOLDERS
MODELS := models

# PROGRAMS AND FLAGS
PYTHON := python3
PYFLAGS := -m
MAIN := competition
MAIN_FLAGS :=
PIP := pip

# COLORS
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# COMMANDS
ECHO := echo -e
MKDIR := mkdir -p
GIT := git
CD := cd
CP := cp

# PARAMETERS
MODEL := classifier
FILE := default
# RULES
.PHONY: help install train test list

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* install		: install the required libraries listed in requirements.txt\n \
	* train		: train and evaluate a specified model (default classifier)\n \
	* test			: test a specified model with the specified weight (default classifier)\n \
	* list			: list available models to train/test\n\n \
	Examples:\n \
	make train MODEL={model}\n \
	make test MODEL={model} FILE={file}"


install:
	@$(ECHO) '$(GREEN)Installing libraries...$(NONE)'
	@$(PIP) install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'


train:
	@$(ECHO) '$(YELLOW)Training and Testing the $(MODEL) model$(NONE)'
	@$(PYTHON) -m competition $(MODEL)
	@$(ECHO) '$(YELLOW)Done$(NONE)'


test:
	@$(ECHO) '$(YELLOW)Testing the $(MODEL) model using $(FILE) weights$(NONE)'
	@$(PYTHON) -m competition $(MODEL) --test $(FILE)
	@$(ECHO) '$(YELLOW)Done$(NONE)'


list:
	@$(ECHO) '$(GREEN)Available models:$(NONE)'
	@$(ECHO) AE
	@$(ECHO) classifier
	@$(ECHO)
	@$(ECHO) Available weights for a model are located in project_root/models/{model_name}/
