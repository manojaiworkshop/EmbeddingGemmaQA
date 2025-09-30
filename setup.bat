@echo off
echo Setting up EmbeddingGemma QA Fine-tuning Environment
echo ================================================

echo.
echo Step 1: Installing Python dependencies...
pip install torch sentence-transformers transformers pandas numpy scikit-learn tqdm matplotlib seaborn

echo.
echo Step 2: Creating sample QA dataset...
python prepare_qa_data.py

echo.
echo Step 3: Setup complete!
echo.
echo Next steps:
echo 1. Run 'python finetune_qa.py' to fine-tune your model
echo 2. Run 'python demo_qa_system.py' to test the QA system
echo 3. Check README.md for detailed instructions

pause
