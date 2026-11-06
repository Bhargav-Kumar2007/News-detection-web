@echo off
setlocal
cd /d "%~dp0"
if not exist models mkdir models
where git >nul 2>nul
if errorlevel 1 (
  echo Git is not installed. Install Git for Windows, then run this file again.
  pause
  exit /b 1
)
if not exist "models\NEWS-detection-using-text\.git" (
  echo Cloning text model repository...
  git clone https://github.com/Bhargav-Kumar2007/NEWS-detection-using-text.git models/NEWS-detection-using-text
  if errorlevel 1 goto failed
) else (
  echo Text model repository already exists.
)
if not exist "models\NEWS-detection-using-image\.git" (
  echo Cloning image model repository...
  git clone https://github.com/Bhargav-Kumar2007/NEWS-detection-using-image.git models/NEWS-detection-using-image
  if errorlevel 1 goto failed
) else (
  echo Image model repository already exists.
)
where git-lfs >nul 2>nul
if not errorlevel 1 (
  git -C models/NEWS-detection-using-image lfs install
  git -C models/NEWS-detection-using-image lfs pull
) else (
  echo Git LFS was not detected. Install it if the checkpoint you need is stored with LFS.
)
if not exist .env copy .env.example .env >nul
if not exist "models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints\best.pt" (
  if exist "models\NEWS-detection-using-image\outputs\recovery\subset_resnet18\checkpoints\best.pt" (
    if not exist "models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints" mkdir "models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints"
    copy "models\NEWS-detection-using-image\outputs\recovery\subset_resnet18\checkpoints\best.pt" "models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints\best.pt" >nul
  )
)
echo.
echo Model repositories are in the models folder.
echo IMPORTANT: the public image repo currently does not include the Fakeddit best.pt checkpoint.
echo Copy your actual trained Fakeddit checkpoint to:
echo models\NEWS-detection-using-image\outputs\subset_resnet18\checkpoints\best.pt
 echo or update IMAGE_CHECKPOINT in .env.
echo Verify TEXT_LABEL_0 and TEXT_LABEL_1 in .env before interpreting a combined score.
echo.
pause
exit /b 0
:failed
echo Failed to clone a model repository. Check your internet connection and GitHub access.
pause
exit /b 1
