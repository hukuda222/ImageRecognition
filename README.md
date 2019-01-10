# ImageRecognition

2018年度 PFNサマーインターンシップの選考課題、chianer/問題2を解いたものです。

https://github.com/pfnet/intern-coding-tasks/tree/master/2018/chainer

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

### ランダムに生成した画像
<img src="https://raw.githubusercontent.com/hukuda222/ImageRecognition/master/result/input_image.png" width="200"/>

### 位置の推定画像
<img src="https://raw.githubusercontent.com/hukuda222/ImageRecognition/master/result/output_image.png" width="200"/>

### 理想的な推定画像
<img src="https://raw.githubusercontent.com/hukuda222/ImageRecognition/master/result/ideal_image.png" width="200"/>
