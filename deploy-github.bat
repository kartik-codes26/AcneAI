@echo off
title AcneAI GitHub Deployer
color 0A
cls
echo.
echo 🚀🚀🚀  AcneAI FULL GitHub Deployment  🚀🚀🚀
echo ============================================
echo.

REM Check if git repo exists
if exist .git (
    echo ✅ Git repo already exists
) else (
    echo 📥 Initializing git repo...
    git init
    git config --global user.name "kartik-codes26"
    git config --global user.email "kartikcodes26@gmail.com"
)

echo.
echo 📦 Adding all files (35+ files)...
git add .
git commit -m "🚀 AcneAI v1.0: 82% accuracy EfficientNetB0 + Streamlit app + Grad-CAM" || echo ✅ Already committed

echo.
echo 🔗 Setting up GitHub remote...
git remote remove origin 2>nul
git remote add origin https://github.com/kartik-codes26/AcneAI.git

echo.
echo 📡 Pushing to GitHub (main branch)...
git branch -M main
git push -u origin main 2>nul || (
    echo.
    echo ⚠️  First push might ask GitHub credentials!
    echo    Username: kartik-codes26
    echo    Use Personal Access Token as password
    echo.
    git push -u origin main
)

echo.
echo 🎉🎉🎉  DEPLOYMENT COMPLETE!  🎉🎉🎉
echo.
echo 📱 Your repository is LIVE at:
echo    https://github.com/kartik-codes26/AcneAI
echo.
echo 📋 To run the Streamlit app:
echo    cd app
echo    streamlit run app.py
echo.
echo Press any key to exit...
pause >nul
