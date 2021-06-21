---
marp: true
theme: mytheme
size: 4:3
paginate: true
footer: "自然言語処理  勉強会  2021/06/21"
---

<!--
headingDivider: 1
-->


# 注意機構 (attention mechanism)

## 概要

- 入力情報の中で特に注目すべき箇所を指定するための機能
	- 画像処理、文字列処理 etc で使用されている
- メリット：
	- 入力の系列情報の冒頭部分を伝播でき、モデルの性能がよくなる
	- (特に画像処理で？) 特定の箇所に着目するので計算コストを抑えれる


## 種類 

- ソフト注意機構
	- 入力情報の重み付き平均を用いる方法
- ハード注意機構
	- 入力情報のどれか一つを確率的に選択して用いる方法
	- 2018年の段階では "hard attention (...略...) is much less widely used" [url](https://arxiv.org/pdf/1808.00300.pdf)
		- この論文では、hard attention ベースの技術で competitive performance を出した
- 自己注意機構（self attention）
	- Transformer 等で使用されている機構
	- どっちに似てる？まったく別？


# ソフト注意機構 

- 系列変換モデルを考える
- 入力系列 $\{x_1,...,x_I\}$、符号化されたベクトル $\{h_1^{(s)},..h_I^{(s)}\}$ として、各時刻の符号化層の隠れ状態ベクトルは

$$
h_i^{(s)} = \Psi^{(s)}\left( x_i, h_{i-1}^{(s)}\right)
$$

- また、復号化器が予測する隠れ状態ベクトルは：

$$
h_j^{(t)} = \Psi^{(t)} (y_j, h_{j-1}^{(t)})
$$

- 通常の系列変換モデルでは encoder → decoder へは $h_I^{(s)}$ のみが渡される
	- $h_I^{(s)}$には全ての情報が含まれているが...もう少し直接的に復号化器に伝播する方法はないか？
	- 入力初期の情報（ex. 文頭の情報）は $\Psi^{(s)}$ が $I$ 回適用されて尚有用な情報として残っておく必要がある

- これらを解決するために、ソフト注意機構を使用することができる

#

- 符号化器の隠れ状態ベクトル $h_i^{(s)}$ の重要度を $a_{ij}$ と定義する
- $a_{ij}$ による重み付き平均は ([Issue #15](https://github.com/mlpnlp/mlpnlp/issues/15))： 
	- 復号化器の隠れ状態 $h_j^{(t)}$ に対して考えるので、$j$ を添える

$$
\bar{h}_j = \sum_{i=1}^I a_{ij}h_{i}^{(s)} 
$$

- 復号化器が$j$番目の単語の予測を行う際に$\bar{h}_j$を利用する
	- （旧版ではconcatがめちゃめちゃなので注意 [Issue #5](https://github.com/mlpnlp/mlpnlp/issues/5)）

$$
\hat{h}_j^{(s)} = \tanh\left(W^{(a)} 
\begin{bmatrix}
\bar{h}\\
h_j^{(t)}
\end{bmatrix}
\right)
$$

- 対象となる $\{h_1^{(s)},...,h_I^{(s)}\}$ の隠れ状態ベクトルの中化kら重要な情報を選別する役割を果たす
	- $\{a_1, ..., a_I\}$もニューラルネットで学習する

- 関数 $\Omega$ で $h_i^{(s)}$ と $h_j^{(t)}$ の重みを計算する
	- $e_i = \Omega\left(h_i^{(s)}, h_j^{(t)}\right)$
	- softmax で１に規格化して確率化する


# ソフト注意機構 (一般化した定義)

- 参照したい情報（符号化器の隠れ状態）を $Y=\{y_1,...,y_N\}$ とする
- 出力に使用する隠れ状態ベクトルを $h_j^{(t)}$ とする
- $h_j^{(t)}$に対してどの$Y$が重要かを $\{a_{1j},...,a_{Nj}\}$で表す
	- この重要度を計算するための情報を $c_{ij}$ とする

$$
a_{ij} = \frac{\exp(\Omega(c_{ij}))}{\sum_{k=1}^N \exp(\Omega(c_{kj}))}
$$

- 復号化器からの出力 $h_j^{(t)}$ と $\hat{y}_j = \sum_{i=1}^N a_{ij}y_i$ を用いて最終的な情報を決定する

## 要するに

- 復号化器の出力にとって符号化器の何が重要かを学習させる
	- 重要度は $a_{ij}$ であり、確率として解釈できる
- 符号化基の隠れ状態ベクトルの期待値をとることで、入力情報の直接的な伝播を実現する


# ハード注意機構

- 参照したい情報 $Y=\{y_1,...,y_N\}$ とする
	- ex. $Y$ は符号化器隠れ状態ベクトル $\{h_1^{(s)},...,h_I^{(s)}\}$
- ハード注意機構では、$Y$ の中から唯一つの情報のみを参照する
	- $P(X=x)=a_x,~x\in \{1,2,...,N\}$ となる確率変数を考える
	- 無作為抽出された $X$ を用いて、使用する $y_i$ の値をただひとつに決定する

$$
\hat{\boldsymbol{y}} = \boldsymbol{y}_x
$$

### 生じる問題

- どの情報を使用するかをひとつに決める→離散的
- 微分不可なので目的関数 $f(\hat{y})$ を微分できず（最小化できず）、backpropagationが使えない
	- そこで期待値を最小化する方法を取る
	- "...but this is still a very active area of research." [link](https://arxiv.org/pdf/1808.00300.pdf)

# 

### 期待値の最小化

- 無作為抽出された $\hat{\boldsymbol{y}}$ を使って目的関数 $f(\hat{\boldsymbol{y}})$ が計算されたとする
	- $\hat{\boldsymbol{y}}$ の実態は $\{\boldsymbol{y}_1,\boldsymbol{y}_2,...,\boldsymbol{y}_N\}$のいずれか
	- 添字は $P(X=x)=a_x,~x\in \{1,2,...,N\}$ を満たす $x$

- 以上を踏まえ目的関数の期待値の勾配は：
$$
\begin{aligned}
\nabla E[f(\hat{y})] & = \nabla \sum_{x=1}^N f(y_x)a_x \\
                     & = \sum_{x=1}^N \nabla f(y_x)a_x + \sum_{x=1}^N f(y_x)a_x \\
                     & = E[\nabla f(\hat{y})] + \sum_{x=1}^N f(y_x) \nabla a_x
\end{aligned}
$$

- いずれの項も$x$の取りうる全ての範囲 $\{1,...,N\}$ に対して計算が必要
	- 何度も $f$ や $\nabla f$の計算をする必要があるので、$N$が大きいと計算が重くなる
	- モンテカルロ法で近似する

# 
### 第一項 $E[\nabla f(\hat{y})]$ の近似

- 確率変数 $X$ を有限個 $T$ だけサンプリングする : $\{\bar{x}_1,...,\bar{x}_T\}$
	- $\{1,...,N\}$ からひとつ取り出す $\times T$ 回
- この標本を用いることで

$$
E[\nabla f(\hat{y})] \simeq \frac{1}{T} \sum_{i=1}^{T} \nabla f(y_{\bar{x}_i})
$$

- モンテカルロ法で近似した期待値は $T\to\infty$ で厳密に真の期待値と一致する
	- (？) 自然言語における$N$と$T$の規模感が分からない...どれくらい？


# 
### 第二項 $\sum_{x=1}^N f(y_x) \nabla a_x$ の近似

- 期待値の形をしていないので、モンテカルロ法をそのままでは適用できない
→期待値の形に変形する
	- 以下の形式まで落とすことでモンテカルロ法を適用できる
$$
\begin{aligned}
\sum_{x=1}^N f(y_x) \nabla a_x &= \sum_{x=1}^{N} f(y_x) \nabla a_x \frac{a_x}{a_x} \\
&= E \left[ f(\hat{y})\frac{\nabla a_x}{a_x} \right] \\
&= E \left[ f(\hat{y}) \nabla \log a_x \right]
\end{aligned}
$$

- 同様に $T$ 個のサンプリングを行うと次のように近似できる

$$
E \left[ f(\hat{y}) \nabla \log a_x \right] \simeq \frac{1}{T} \sum_{i=1}^T f(y_{\bar{x}_i}) \nabla \log a_{\bar{x}_i} 
$$

# ソフトとハード注意機構の違い


### 文字の整理

- 参照したい情報 $Y= \{ \boldsymbol{y}_1,..., \boldsymbol{y}_N \}$
	- 実際に参照するのは、とある変形を施した情報の $\hat{\boldsymbol{y}}$
- $Y$ に対しての重み (=確率)は $\{a_1,...,a_N\}$
- 目的関数 $f(\cdot)$

### 違いの羅列

- ソフト注意機構
	- $\hat{\boldsymbol{y}}=E[ \boldsymbol{y}]$
	- 目的関数は $f(\hat{y})$ 
	- 最適化の対象は、そのままこの目的関数
		- $f$ への入力は確定した値なので、$f$ の計算は一回で済む

- ハード注意機構
	- $\hat{\boldsymbol{y}}=\boldsymbol{y}_x$
	- 目的関数は $f(\hat{y})$ 
	- 最適化の対象は、目的関数の期待値 $E[ f(\hat{y})]$
		- $f$ への入力は確率変数なので、$f$ の計算は複数回必要→MCで近似


# ハード注意機構の学習における問題点

- MCで近似したときの $f(y_{\bar{x}_i}) \log a_{\bar{x}_i}$ の分散が大きくてうまく学習できないことがあることが知られている

- 分散を抑える工夫を、定数$b$を用いて議論する

$$
\begin{aligned}
E[ f(\hat{y}) \nabla \log a_{x} ] &= E[ f(\hat{y}) \nabla \log a_x -b\nabla\log a_x + b\nabla\log a_x] \\
&= E[ (f(\hat{y}) -b)\nabla\log a_x + b\nabla\log a_x] \\
&= E[ (f(\hat{y}) -b)\nabla\log a_x] + bE[\nabla\log a_x] \\
\end{aligned}
$$

- ここで $\log a_x$ が従う確率分布が $a_x$ であることに注意すると
	- あと $\sum a_x=1$ も思い出す

$$
E[\nabla \log a_x] = \sum_{x=1}^N \nabla \log a_x \times a_x = \sum_{x=1}^N\frac{\nabla a_x}{a_x} a_x = \sum_{x=1}^N \nabla a_x = \nabla \sum_{x=1}^N a_x  = 0
$$

- 以上を用いて

$$
E[f(\hat{y})\nabla \log a_x] = E[ (f(\hat{y}) -b)\nabla\log a_x]
$$

- 分散が小さくなるように $b$ を設定できれば推定値の精度が上がる
	- $b=E[ f(\hat{y})]$を使用することが多いとのこと


# その他の注意機構

...割愛... 

# 記憶ネットワーク (memory networks) とは

## 概要
- より直接的に記憶の仕組みをモデル化することができる
	- LSTMを始めとするRNNでは記憶の内容（隠れ状態ベクトル）は固定長で限定的だった
	- 知識を蓄えておいて、質問に対して応答する
	- 出力を生成するための情報（知識情報）を与える点が特徴的

- 注意点：「入力」は二種類の意味で使われる
	- 知識を蓄える、いわゆる"インプット"という意味での「入力」
	- 質問をモデルに入れる、という意味での「入力」

## 記憶ネットワークのモデル

- 記憶ネットワークは内部に $N$ 個の記憶情報 $\boldsymbol{M}=(\boldsymbol{m}_i)_{i=1}^N$ を列として持つ
- ネットワーク内部は4つの部品に分解して考えることができる
	- 入力情報変換：入力された情報を内部表現に変換する
	- 一般化：新しい知識源の情報を利用して、内部の記憶情報 $M$ を更新する
	- 出力情報変換：質問に対して返答のための内部表現を生成する
	- 応答：出力情報を適切なフォーマットに変換する


# 記憶ネットワークの種類

## 前提 
- 根拠情報 (supporting fact) を使用するかどうか
	- 最終的な出力（返答）を生成する際の根拠
	- どんな知識を組み合わせて、この出力（返答）に至ったかを学習の段階で使用できるか
		- ex. 対話ログのように、質問に対して返答のみが記録されている訓練データでは、根拠情報は欠けていると言える（どうしてその返答に至ったかは、本データからは定かではない）
	- 「中間的な部分課題の解」とも教科書では言い換えられている

## 教科書で議論する種類
- ３種類の記憶ネットワーク
	- 根拠情報が与えられた条件下で学習を行う：強教師あり記憶ネットワーク
	- 根拠情報を使用しない学習を行う：end-to-end 記憶ネットワーク
	- 返答に使用する知識を繰り返し問い合わせるモデル：動的記憶ネットワーク

# 教師あり記憶ネットワーク

## 知識を蓄えるための "入力"
- 知識を蓄える
	- 記憶として蓄える文章（ここでは単語列）を $\boldsymbol{x}$ とおく
		- 単語はone-hotベクトルになっていて、$\boldsymbol{x}_j$ で $j$ 番目の単語を示すとする
		- 単語の次元数は語彙数 $|\mathcal{V}|$ 
	- $D\times \mathcal{V}$ の埋め込み行列 $E$ を用いて次のように入力情報変換を行う

$$
I(\boldsymbol{x}) = \sum_{j\in \mathcal{V}} E \boldsymbol{x}_j
$$

- 式(4.20) はこれ以降この節では登場しないが...察するに（cf.新版では$I(x)$は削除されている）
	- 知識を蓄えるための文章が複数あって、その一文を $i$ で指定すると
	  $m_i \leftarrow I(\boldsymbol{x}_i)$ ということを言ってる（cf. 式(4.26)）
	- $i=1,...,N$ （$N$ 個の"入力"文）を使用することで、計$N$個の記憶情報を追加することになる

$$
I(\boldsymbol{x}_i) = \sum_{j\in \mathcal{V}} E \boldsymbol{x}_{ij}
$$

# 教師あり記憶ネットワーク

## モデルから返答を引き出すための "入力"

- さきの $\boldsymbol{x}$ の "入力" とは意味が違うと思う
	- 式(4.21)以降の入力 $\boldsymbol{x}$ は、質問文としての入力（ちゃんとした文章）

- 質問文への返答に必要な記憶かどうかを判定するスコア関数 $s_O$ を用意する
	- これを使ってスコア上位 $\kappa$ 個の知識を取ってくる (例では $\kappa=2$)
	- 質問文 $\boldsymbol{x}$ に対して最も関係の深い知識情報を $\boldsymbol{m}_{o1}$とする

$$
\begin{aligned}
o_1 &= O_1(\boldsymbol{x}, M) = \mathop{\text{argmax}}\limits_{i=1,...,N} s_O (\boldsymbol{x}, \boldsymbol{m}_i)\\
o_2 &= O_2(\boldsymbol{x}, M) = \mathop{\text{argmax}}\limits_{i=1,...,N} s_O ((\boldsymbol{x},\boldsymbol{m}_{o1}), \boldsymbol{m}_i)
\end{aligned}
$$

- 質問文に対して関係の深い知識を取ってこれたら、返答を生成する
	- 語彙集合 $\mathcal{V}$ の中からスコア $s_R$ が最大となる単語を探し、それを出力とする

$$
r = \mathop{\text{argmax}}\limits_{w\in \mathcal{V}} s_R ( (\boldsymbol{x}, \boldsymbol{m}_{o1}, \boldsymbol{m}_{o2}), w)
$$

# end-to-end 記憶ネットワーク

# 動的記憶ネットワーク

- Dynamic memory networks (DMN)
	- 入力
	- 意味記憶
	- 質問
	- エピソード記憶
	- 回答


# §4.3 出力層の高速化




