# Prerequisite
## Download stylegan2-ada-pytorch
In next step, we can configure **stylegan2-ada-pytorch** as submodule of our repo. Then you don't need to manually download **stylegan2-ada-pytorch** or upload it into our repo.
```sh
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

## Install Visual Studio
* Download and install Visual Studio from the official Microsoft website: https://visualstudio.microsoft.com/downloads/

* Install C++ build tool: Open **Visual Studio 2022 Installer** -> Choose **Desktop development with C++** -> Install

* Add MSVC to Environment Variable Path: refer to [guide](https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)#to-add-a-path-to-the-path-environment-variable). The path for visual studio 2022 msvc is: **C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.36.32532\bin\Hostx64\x64**

## Install dlib
```powershell
wget http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2

# Install choco
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install bzip2
choco install bzip2

# Install dlib dependency
bzip2 -d shape_predictor_5_face_landmarks.dat.bz2
```

# Install python packages
```sh
# pip install packages
pip install opencv-python
pip install pillow
pip install dlib
pip install matplotlib
pip install click
pip install imageio
pip install ninja
pip install imageio[ffmpeg]
pip install imageio[pyav]
# generate torch install by https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Test
```sh
python ./image_to_video.py
```
