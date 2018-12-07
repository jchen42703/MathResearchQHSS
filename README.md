# Math Research QHSS
Relevant statistics and machine learning procedures for math research in QHSS@YC

## Statistics Tests
* Chi-Square`[]`
* Fisher's Exact`[x]`
* Event Study`[x]`
  * Multiple Events Concurrently`[]`
* ARIMA`[]`

## Basic Machine Learning
* __Cyclical Learning Rates for Visual Speech Recognition using Deep Learning__
  * In the `/lipreading/` repository
    * New dependencies:
     * sk-video, ffmpeg
  * Uses [`keras`](https://keras.io/) with the [`tensorflow`](https://www.tensorflow.org/) backend to build CNNs + LSTMs for lipreading on the [GRID Corpus dataset](http://spandh.dcs.shef.ac.uk/gridcorpus/) with no feature extraction.
    * some code inspired by https://github.com/rizkiarm/LipNet/tree/master/ 
    * CLR is built from a [fork](https://github.com/jchen42703/CLR) of [this CLR repository](https://github.com/bckenstler/CLR)
* Preprocessing`[]` 
  * Images
  * Tabular
  * Sequential
* Building models with Keras

## Installation
```
git clone https://github.com/jchen42703/MathResearchQHSS.git
cd MathResearchQHSS
pip install -r requirements.txt
cd '/home/'
python import MathResearchQHSS
```
To address the addition of `/lipreading`: 
```
apt install ffmpeg
pip install ffmpeg sk-video
```
