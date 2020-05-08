# csci5980
Final project for CSci 5980: deep learning for automatic music translation.

Follow theses steps to install all package dependencies for running the model:

We first install software dependencies for manipulating raw audio (``ffmpeg``):

1. Create a local software directory
  `mkdir ~/software`

2. Install the NASM assembler (dependency of ffmpeg):
  ```bash
  cd ~/software
  wget https://www.nasm.us/pub/nasm/releasebuilds/2.14.02/nasm-2.14.02.tar.bz2
  tar -xvf nasm-2.14.02.tar.bz2
  cd nasm-2.14.02
  ./configure --prefix=~/software/nasm/
  make install
  export PATH=$PATH:~/software/nasm/bin/
  ```

3. Make sure that NASM assembler installed correctly:
  ```bash
  nasm -v
  ```
  The output should look something like:
  `NASM version 2.14.02 compiled on Mar 11 2020`

4. Install ffmpeg:
  ```bash
  cd ~/software
  wget https://ffmpeg.org/releases/ffmpeg-4.2.2.tar.bz2
  tar -xvf ffmpeg-4.2.2.tar.bz2
  cd ffmpeg-4.2.2
  ./configure --prefix=~/software/ffmpeg/
  make install
  export PATH=$PATH:~/software/ffmpeg/bin/
  ```

5. Make sure that ffmpeg installed correctly:
  ```bash
  ffmpeg -version
  ```
  The output should look something like:
  ```
  ffmpeg version 4.2.2 Copyright (c) 2000-2019 the FFmpeg developers
  built with gcc 4.4.7 (GCC) 20120313 (Red Hat 4.4.7-23)
  configuration: --prefix=/home/csci5980/piehl008/software/ffmpeg/
  libavutil      56. 31.100 / 56. 31.100
  libavcodec     58. 54.100 / 58. 54.100
  libavformat    58. 29.100 / 58. 29.100
  libavdevice    58.  8.100 / 58.  8.100
  libavfilter     7. 57.100 /  7. 57.100
  libswscale      5.  5.100 /  5.  5.100
  libswresample   3.  5.100 /  3.  5.100
  ```

6. Now, we can make the virtual environment and install python packages.  First, create the virtual environment by running:

`conda create --name audio-proj python=3.7`

7. Next, install packages by running

```bash
cd ~/csci5980
conda install --name audio-proj --file requirements.txt --channel defaults --channel conda-forge
```

(Note: this can take a while - and you need to say yes to installing everything after it solves the environment)

8. To activate the virtual environment, you can now run `source activate audio-proj`. Note: you should do this to test that you can activate the virtual evironment, but you probably shouldn't run a lot unless you are submitting jobs to the queue.  If you want to use this virtual environment through the MSI notebooks, check out the tutorial at https://sunju.org/teach/DL-Spring-2020/TensorFlowPyTorch.html.

### Adding the Virtual Environment to Jupyter Notebooks

Now that we have created the virtual environment, we can add it to the Jupyter notebook kernels so that we can use the virtual environment through MSI's notebook server. To do this, we have to add the kernel specifications to the known Jupyter kernels for our user:

9. If you haven't already, activate your virtual environment by running `source activate audio-proj`. Then enter

```bash 
which python
```

Your output should tell you where the python executable for this virtual environment lives - the output for me displays `~/.conda/envs/audio-proj/bin/python`.  If you see something that looks like `/panfs/roc/msisoft/anaconda/anaconda3-2018.12/bin/python`, go back and make sure that you have the virtual environment active and try again. After you have an ouput that clearly has the name of the virtual environment in the directory path (i.e. contains audio-proj in it), continue to the next step.

10. Now, we need to create the kernel configuration. To do this run

```bash
mkdir ~/.local/share/jupyter/kernels/audio-proj
nano ~/.local/share/jupyter/kernels/audio-proj/kernel.json
```

The nano command will open a very basic text editor that you can navigate with the arrow keys. Enter the following:

```text
{
 "argv": [
  "~/.conda/envs/audio-proj/bin/python", #replace this with your path from step 9 above! (and delete this comment)
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "Audio Project Kernel",
 "language": "python"
}
```
where you replace the first line of the argv array with whatever executable path was output from step 9 above (it likely will be identical to this). To exit the nano text editor, type `Ctrl-x <RETURN>` and then type `Y <RETURN>` to save the file.

11. Now that you have saved the kernel file, you should be able to go to `https://notebooks.msi.umn.edu/` and when you click on the `New` tab to create a new file, you should be able to select `Audio Project Kernel` as an available kernel to run your newly created file in.