@echo off
REM ============================================================
REM experiment.bat — Full Vietnamese TSR Pipeline (Windows)
REM Runs: data check -> scratch training -> pretrained training
REM       -> evaluation comparison -> pruning
REM
REM Usage: experiment.bat [epochs] [batch]
REM   Default: epochs=100, batch=16
REM ============================================================

SET "ROOT=%~dp0"
CD /D "%ROOT%"

SET EPOCHS=%1
IF "%EPOCHS%"=="" SET EPOCHS=100

SET BATCH=%2
IF "%BATCH%"=="" SET BATCH=16

ECHO ========================================
ECHO   Vietnamese TSR --- Full Pipeline
ECHO   Epochs : %EPOCHS%  ^|  Batch : %BATCH%
ECHO ========================================

REM ── Step 1: Validate dataset ────────────────────────────────
ECHO.
ECHO [1/5] Validating dataset ...
python data/data_loader.py
IF ERRORLEVEL 1 ( ECHO [ERROR] Dataset validation failed. & EXIT /B 1 )

REM ── Step 2: Train from scratch ──────────────────────────────
ECHO.
ECHO [2/5] Training from SCRATCH ...
python training/trainer.py --mode scratch --epochs %EPOCHS% --batch %BATCH%
IF ERRORLEVEL 1 ( ECHO [ERROR] Scratch training failed. & EXIT /B 1 )

REM ── Step 3: Train pretrained (fine-tune) ────────────────────
ECHO.
ECHO [3/5] Training PRETRAINED (fine-tune) ...
python training/trainer.py --mode pretrained --epochs %EPOCHS% --batch %BATCH%
IF ERRORLEVEL 1 ( ECHO [ERROR] Pretrained training failed. & EXIT /B 1 )

REM ── Step 4: Evaluate both + compare ─────────────────────────
ECHO.
ECHO [4/5] Evaluating and comparing both modes ...
python training/evaluator.py --compare
IF ERRORLEVEL 1 ( ECHO [ERROR] Evaluation failed. & EXIT /B 1 )

REM ── Step 5: Prune pretrained model ──────────────────────────
ECHO.
ECHO [5/5] Pruning pretrained model (sparsity=30%%) ...
python pruning/baseline.py --mode pretrained --sparsity 0.3
IF ERRORLEVEL 1 ( ECHO [ERROR] Pruning failed. & EXIT /B 1 )

ECHO.
ECHO ========================================
ECHO   Pipeline complete!  Results in EXPERIMENT/
ECHO ========================================
PAUSE
