#! /bin/bash
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d ..
mv ../chest_xray ../datasets
rm chest-xray-pneumonia.zip
