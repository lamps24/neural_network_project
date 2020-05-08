Bootstrap: docker
From: tensorflow/tensorflow:latest-py3

Name:		csci5980_audio_proj


%configure
make %{?_smp_mflags}


%install
rm -rf $RPM_BUILD_ROOT
make install DESTDIR=$RPM_BUILD_ROOT


%clean
rm -rf $RPM_BUILD_ROOT


%files
train.py
train.sh
requirements.txt
utils/
network/
data/


%environment
export PYTHONPATH=$PYTHONPATH:/:/utils/


%post
apt-get update
apt-get -y install ffmpeg
python3 -m pip install -r requirements.txt
