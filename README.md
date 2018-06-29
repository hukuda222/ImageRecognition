# ImageRecognition
画像に埋め込まれている別の画像を検出するものです、
2つの画像セットから、画像の一部に別の画像を埋め込んだものを学習データとしています。

以下のように実行します。例えば、./DATAのパスにaaa.png,bbb.png,...といった画像がある場合は、./DATA.*.pngを画像セットのパスとして使ってください。
学習終了後にmodel.npzを出力します。
```
python train.py --path0 {"読み込みたい画像セットAのパス"} --path1 {"読み込みたい画像セットBのパス"}
```

以下のように実行することで、画像セットAの中のある画像に、画像セットBの中のある画像を埋め込んだ画像をのどこに画像が埋め込まれているかを示す画像を出力します。
```
python train.py --path0 {"読み込みたい画像セットAのパス"} --path1 {"読み込みたい画像セットBのパス"} --image
```