@echo off
echo 🚀 Setting up AcneAI GitHub repository...
echo.

REM Initialize git repo
git init
git config --global user.name "kartik-codes26"
git config --global user.email "kartikcodes26@gmail.com"
git add .
git commit -m "🚀 Initial AcneAI commit: 82% accuracy model + Streamlit app + Grad-CAM"

echo.
echo ✅ Git repository initialized and first commit created!
echo.
echo 📋 NEXT STEPS:
echo 1. Go to https://github.com/new
echo 2. Create repo named "AcneAI" (Public, NO README)
echo 3. Run these commands:
echo    git remote add origin https://github.com/kartik-codes26/AcneAI.git
echo    git branch -M main                                   
echo    git push -u origin main
echo.
echo Your repo will be live at: https://github.com/kartik-codes26/AcneAI
pause
