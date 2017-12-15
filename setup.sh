#!/bin/bash
apt-get install scons libsdl1.2-dev subversion libgtk2.0-dev xvfb liblua
cd fceux-2.2.2
sudo scons install

virtualenv .env
source .env/bin/activate
python setup.py
pip install -r requirements.txt
