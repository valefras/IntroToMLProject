# FOLDERS
MODELS := models

# PROGRAMS AND FLAGS
PYTHON := python3
PYFLAGS := -m
MAIN := competition
MAIN_FLAGS :=
PIP := pip

# ======= TRAIN =========
TRAIN :=
TRAIN_FLAGS :=

# ======= TEST  =========
TEST :=
TEST_FLAGS :=

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

# RULES
.PHONY: help install

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* install		: install the required libraries listed in requirements.txt"


install:
	@$(ECHO) '$(GREEN)Installing libraries...$(NONE)'
	@$(PIP) install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'
