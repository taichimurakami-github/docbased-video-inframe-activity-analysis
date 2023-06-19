# Document-based video inframe events analysis

## イベントの定義

### 1. Drawing

- ビデオ内の黒板の色を背景とする．背景ではないピクセル（non-background pixel）の量が，連続するフレームの間で，目覚ましい増加が検知されるまで比較し続けることで，これを Drawing の開始地点と判別できる．
- 教師が何かを描き続ける限り，このような顕著な non-background pixel の増加が継続するはずである．描くのを辞めると，この増加は止まる．この時点を Drawing の終了地点と判別できる．

### 2. Erasing

### 3. Animation

- 主にスライド発生

### 4. Embedded video
