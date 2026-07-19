# JKB v1 ソースマトリクス（オーケストレーター設計、sonnet5 バッチ委任用）

各セル (domain × difficulty) につき Wikipedia-ja 種記事群と難度定義を明示。sonnet5 は各セル 14 問を、この URL 群だけをソースとして著述する（外部知識に頼らない・URL に載っていない事実は書かない）。

出力形式は `datasets/eval/jkb/schema.md` の JSONL 定義に厳格に従う。1 バッチ = 1 セル = ~14 問。

---

## 地理 (geo)

### core-14
記事群 (日本人ならほぼ全員知る山川湖県):
- https://ja.wikipedia.org/wiki/富士山
- https://ja.wikipedia.org/wiki/信濃川
- https://ja.wikipedia.org/wiki/琵琶湖
- https://ja.wikipedia.org/wiki/北海道
- https://ja.wikipedia.org/wiki/沖縄県
- https://ja.wikipedia.org/wiki/東京都
- https://ja.wikipedia.org/wiki/日本の地理
- https://ja.wikipedia.org/wiki/瀬戸内海
- https://ja.wikipedia.org/wiki/日本海
- https://ja.wikipedia.org/wiki/太平洋
- https://ja.wikipedia.org/wiki/淀川
- https://ja.wikipedia.org/wiki/利根川

### standard-14
記事群 (義務教育レベル。山川湖の 2 番手・地方名):
- https://ja.wikipedia.org/wiki/北岳
- https://ja.wikipedia.org/wiki/奥穂高岳
- https://ja.wikipedia.org/wiki/石狩川
- https://ja.wikipedia.org/wiki/最上川
- https://ja.wikipedia.org/wiki/十和田湖
- https://ja.wikipedia.org/wiki/霞ヶ浦
- https://ja.wikipedia.org/wiki/日本の地方区分
- https://ja.wikipedia.org/wiki/中部地方
- https://ja.wikipedia.org/wiki/近畿地方
- https://ja.wikipedia.org/wiki/九州地方
- https://ja.wikipedia.org/wiki/中国地方
- https://ja.wikipedia.org/wiki/四国
- https://ja.wikipedia.org/wiki/東北地方

### advanced-14
記事群 (旧国名・地形の細目):
- https://ja.wikipedia.org/wiki/令制国
- https://ja.wikipedia.org/wiki/五畿七道
- https://ja.wikipedia.org/wiki/陸奥国
- https://ja.wikipedia.org/wiki/薩摩国
- https://ja.wikipedia.org/wiki/近江国
- https://ja.wikipedia.org/wiki/日本の活火山一覧
- https://ja.wikipedia.org/wiki/フォッサマグナ
- https://ja.wikipedia.org/wiki/中央構造線
- https://ja.wikipedia.org/wiki/対馬海流
- https://ja.wikipedia.org/wiki/黒潮
- https://ja.wikipedia.org/wiki/親潮
- https://ja.wikipedia.org/wiki/糸魚川静岡構造線

---

## 歴史 (hist)

### core-14
- https://ja.wikipedia.org/wiki/徳川家康
- https://ja.wikipedia.org/wiki/織田信長
- https://ja.wikipedia.org/wiki/豊臣秀吉
- https://ja.wikipedia.org/wiki/源頼朝
- https://ja.wikipedia.org/wiki/明治維新
- https://ja.wikipedia.org/wiki/聖徳太子
- https://ja.wikipedia.org/wiki/大化の改新
- https://ja.wikipedia.org/wiki/江戸幕府
- https://ja.wikipedia.org/wiki/鎌倉幕府
- https://ja.wikipedia.org/wiki/坂本龍馬
- https://ja.wikipedia.org/wiki/西郷隆盛
- https://ja.wikipedia.org/wiki/太平洋戦争

### standard-14
- https://ja.wikipedia.org/wiki/桓武天皇
- https://ja.wikipedia.org/wiki/平安京
- https://ja.wikipedia.org/wiki/藤原道長
- https://ja.wikipedia.org/wiki/後醍醐天皇
- https://ja.wikipedia.org/wiki/建武の新政
- https://ja.wikipedia.org/wiki/応仁の乱
- https://ja.wikipedia.org/wiki/関ヶ原の戦い
- https://ja.wikipedia.org/wiki/大政奉還
- https://ja.wikipedia.org/wiki/廃藩置県
- https://ja.wikipedia.org/wiki/日露戦争
- https://ja.wikipedia.org/wiki/日清戦争
- https://ja.wikipedia.org/wiki/五箇条の御誓文

### advanced-14
- https://ja.wikipedia.org/wiki/延暦
- https://ja.wikipedia.org/wiki/仁徳天皇陵
- https://ja.wikipedia.org/wiki/古事記
- https://ja.wikipedia.org/wiki/日本書紀
- https://ja.wikipedia.org/wiki/白村江の戦い
- https://ja.wikipedia.org/wiki/承久の乱
- https://ja.wikipedia.org/wiki/建武式目
- https://ja.wikipedia.org/wiki/南北朝時代_(日本)
- https://ja.wikipedia.org/wiki/享保の改革
- https://ja.wikipedia.org/wiki/寛政の改革
- https://ja.wikipedia.org/wiki/天保の改革
- https://ja.wikipedia.org/wiki/王政復古_(日本)

---

## 文学 (lit)

### core-14
- https://ja.wikipedia.org/wiki/源氏物語
- https://ja.wikipedia.org/wiki/紫式部
- https://ja.wikipedia.org/wiki/清少納言
- https://ja.wikipedia.org/wiki/枕草子
- https://ja.wikipedia.org/wiki/夏目漱石
- https://ja.wikipedia.org/wiki/吾輩は猫である
- https://ja.wikipedia.org/wiki/川端康成
- https://ja.wikipedia.org/wiki/雪国_(小説)
- https://ja.wikipedia.org/wiki/芥川龍之介
- https://ja.wikipedia.org/wiki/羅生門_(小説)
- https://ja.wikipedia.org/wiki/太宰治
- https://ja.wikipedia.org/wiki/走れメロス

### standard-14
- https://ja.wikipedia.org/wiki/松尾芭蕉
- https://ja.wikipedia.org/wiki/奥の細道
- https://ja.wikipedia.org/wiki/森鴎外
- https://ja.wikipedia.org/wiki/舞姫_(森鴎外)
- https://ja.wikipedia.org/wiki/宮沢賢治
- https://ja.wikipedia.org/wiki/銀河鉄道の夜
- https://ja.wikipedia.org/wiki/三島由紀夫
- https://ja.wikipedia.org/wiki/金閣寺_(小説)
- https://ja.wikipedia.org/wiki/大江健三郎
- https://ja.wikipedia.org/wiki/村上春樹
- https://ja.wikipedia.org/wiki/樋口一葉
- https://ja.wikipedia.org/wiki/伊豆の踊子

### advanced-14
- https://ja.wikipedia.org/wiki/竹取物語
- https://ja.wikipedia.org/wiki/伊勢物語
- https://ja.wikipedia.org/wiki/土佐日記
- https://ja.wikipedia.org/wiki/更級日記
- https://ja.wikipedia.org/wiki/新古今和歌集
- https://ja.wikipedia.org/wiki/古今和歌集
- https://ja.wikipedia.org/wiki/万葉集
- https://ja.wikipedia.org/wiki/柿本人麻呂
- https://ja.wikipedia.org/wiki/在原業平
- https://ja.wikipedia.org/wiki/西行
- https://ja.wikipedia.org/wiki/井原西鶴
- https://ja.wikipedia.org/wiki/近松門左衛門

---

## 食文化 (food)

### core-14
- https://ja.wikipedia.org/wiki/寿司
- https://ja.wikipedia.org/wiki/天ぷら
- https://ja.wikipedia.org/wiki/味噌
- https://ja.wikipedia.org/wiki/醤油
- https://ja.wikipedia.org/wiki/日本酒
- https://ja.wikipedia.org/wiki/そば
- https://ja.wikipedia.org/wiki/うどん
- https://ja.wikipedia.org/wiki/ラーメン
- https://ja.wikipedia.org/wiki/梅干し
- https://ja.wikipedia.org/wiki/納豆
- https://ja.wikipedia.org/wiki/だし
- https://ja.wikipedia.org/wiki/昆布

### standard-14
- https://ja.wikipedia.org/wiki/懐石料理
- https://ja.wikipedia.org/wiki/精進料理
- https://ja.wikipedia.org/wiki/雑煮
- https://ja.wikipedia.org/wiki/おせち料理
- https://ja.wikipedia.org/wiki/柏餅
- https://ja.wikipedia.org/wiki/ちらし寿司
- https://ja.wikipedia.org/wiki/焼酎
- https://ja.wikipedia.org/wiki/みりん
- https://ja.wikipedia.org/wiki/かつお節
- https://ja.wikipedia.org/wiki/わさび
- https://ja.wikipedia.org/wiki/漬物
- https://ja.wikipedia.org/wiki/牛丼

### advanced-14
- https://ja.wikipedia.org/wiki/一汁三菜
- https://ja.wikipedia.org/wiki/五節句
- https://ja.wikipedia.org/wiki/京料理
- https://ja.wikipedia.org/wiki/加賀料理
- https://ja.wikipedia.org/wiki/沖縄料理
- https://ja.wikipedia.org/wiki/へしこ
- https://ja.wikipedia.org/wiki/ふなずし
- https://ja.wikipedia.org/wiki/柿の葉寿司
- https://ja.wikipedia.org/wiki/釜飯
- https://ja.wikipedia.org/wiki/ずんだ
- https://ja.wikipedia.org/wiki/しもつかれ
- https://ja.wikipedia.org/wiki/ほうとう

---

## 伝統文化 (trad)

### core-14
- https://ja.wikipedia.org/wiki/歌舞伎
- https://ja.wikipedia.org/wiki/能
- https://ja.wikipedia.org/wiki/大相撲
- https://ja.wikipedia.org/wiki/茶道
- https://ja.wikipedia.org/wiki/華道
- https://ja.wikipedia.org/wiki/柔道
- https://ja.wikipedia.org/wiki/剣道
- https://ja.wikipedia.org/wiki/祇園祭
- https://ja.wikipedia.org/wiki/七夕
- https://ja.wikipedia.org/wiki/端午の節句
- https://ja.wikipedia.org/wiki/雛祭り
- https://ja.wikipedia.org/wiki/正月

### standard-14
- https://ja.wikipedia.org/wiki/文楽
- https://ja.wikipedia.org/wiki/狂言
- https://ja.wikipedia.org/wiki/世阿弥
- https://ja.wikipedia.org/wiki/千利休
- https://ja.wikipedia.org/wiki/浮世絵
- https://ja.wikipedia.org/wiki/葛飾北斎
- https://ja.wikipedia.org/wiki/歌川広重
- https://ja.wikipedia.org/wiki/俳句
- https://ja.wikipedia.org/wiki/生け花
- https://ja.wikipedia.org/wiki/合気道
- https://ja.wikipedia.org/wiki/相撲部屋
- https://ja.wikipedia.org/wiki/横綱

### advanced-14
- https://ja.wikipedia.org/wiki/黒田節
- https://ja.wikipedia.org/wiki/よさこい祭り
- https://ja.wikipedia.org/wiki/ねぶた祭
- https://ja.wikipedia.org/wiki/竿燈
- https://ja.wikipedia.org/wiki/秋田竿燈まつり
- https://ja.wikipedia.org/wiki/天神祭
- https://ja.wikipedia.org/wiki/神田祭
- https://ja.wikipedia.org/wiki/三大祭_(日本)
- https://ja.wikipedia.org/wiki/雅楽
- https://ja.wikipedia.org/wiki/箏
- https://ja.wikipedia.org/wiki/三味線
- https://ja.wikipedia.org/wiki/尺八

---

## 政治・制度 (pol)

### core-14
- https://ja.wikipedia.org/wiki/日本国憲法
- https://ja.wikipedia.org/wiki/内閣総理大臣
- https://ja.wikipedia.org/wiki/国会_(日本)
- https://ja.wikipedia.org/wiki/衆議院
- https://ja.wikipedia.org/wiki/参議院
- https://ja.wikipedia.org/wiki/天皇
- https://ja.wikipedia.org/wiki/日本国憲法第9条
- https://ja.wikipedia.org/wiki/最高裁判所
- https://ja.wikipedia.org/wiki/日本の都道府県
- https://ja.wikipedia.org/wiki/選挙権
- https://ja.wikipedia.org/wiki/国旗
- https://ja.wikipedia.org/wiki/日章旗

### standard-14
- https://ja.wikipedia.org/wiki/大日本帝国憲法
- https://ja.wikipedia.org/wiki/内閣_(日本)
- https://ja.wikipedia.org/wiki/内閣官房長官
- https://ja.wikipedia.org/wiki/財務省_(日本)
- https://ja.wikipedia.org/wiki/外務省
- https://ja.wikipedia.org/wiki/文部科学省
- https://ja.wikipedia.org/wiki/参議院議員通常選挙
- https://ja.wikipedia.org/wiki/衆議院議員総選挙
- https://ja.wikipedia.org/wiki/被選挙権
- https://ja.wikipedia.org/wiki/君が代
- https://ja.wikipedia.org/wiki/国事行為
- https://ja.wikipedia.org/wiki/日本国憲法第25条

### advanced-14
- https://ja.wikipedia.org/wiki/五・一五事件
- https://ja.wikipedia.org/wiki/二・二六事件
- https://ja.wikipedia.org/wiki/普通選挙法
- https://ja.wikipedia.org/wiki/治安維持法
- https://ja.wikipedia.org/wiki/日本国憲法第14条
- https://ja.wikipedia.org/wiki/日本国憲法第96条
- https://ja.wikipedia.org/wiki/衆議院解散
- https://ja.wikipedia.org/wiki/内閣総理大臣指名選挙
- https://ja.wikipedia.org/wiki/両院協議会
- https://ja.wikipedia.org/wiki/違憲審査制
- https://ja.wikipedia.org/wiki/裁判員制度
- https://ja.wikipedia.org/wiki/検察審査会

---

## 生活・慣習 (life)

### core-14
- https://ja.wikipedia.org/wiki/日本の祝日
- https://ja.wikipedia.org/wiki/元日
- https://ja.wikipedia.org/wiki/成人の日
- https://ja.wikipedia.org/wiki/建国記念の日
- https://ja.wikipedia.org/wiki/こどもの日
- https://ja.wikipedia.org/wiki/敬老の日
- https://ja.wikipedia.org/wiki/勤労感謝の日
- https://ja.wikipedia.org/wiki/文化の日
- https://ja.wikipedia.org/wiki/お盆
- https://ja.wikipedia.org/wiki/年賀状
- https://ja.wikipedia.org/wiki/お年玉
- https://ja.wikipedia.org/wiki/成人式

### standard-14
- https://ja.wikipedia.org/wiki/戸籍
- https://ja.wikipedia.org/wiki/住民票
- https://ja.wikipedia.org/wiki/マイナンバー
- https://ja.wikipedia.org/wiki/国民健康保険
- https://ja.wikipedia.org/wiki/健康保険
- https://ja.wikipedia.org/wiki/国民年金
- https://ja.wikipedia.org/wiki/厚生年金保険
- https://ja.wikipedia.org/wiki/介護保険
- https://ja.wikipedia.org/wiki/確定申告
- https://ja.wikipedia.org/wiki/所得税
- https://ja.wikipedia.org/wiki/消費税
- https://ja.wikipedia.org/wiki/敬語

### advanced-14
- https://ja.wikipedia.org/wiki/冠婚葬祭
- https://ja.wikipedia.org/wiki/結納
- https://ja.wikipedia.org/wiki/初七日
- https://ja.wikipedia.org/wiki/四十九日
- https://ja.wikipedia.org/wiki/一周忌
- https://ja.wikipedia.org/wiki/新盆
- https://ja.wikipedia.org/wiki/お彼岸
- https://ja.wikipedia.org/wiki/七五三
- https://ja.wikipedia.org/wiki/還暦
- https://ja.wikipedia.org/wiki/古希
- https://ja.wikipedia.org/wiki/喜寿
- https://ja.wikipedia.org/wiki/傘寿

---

## 地域・観光 (region)

### core-14
- https://ja.wikipedia.org/wiki/京都
- https://ja.wikipedia.org/wiki/奈良
- https://ja.wikipedia.org/wiki/大阪
- https://ja.wikipedia.org/wiki/横浜
- https://ja.wikipedia.org/wiki/札幌
- https://ja.wikipedia.org/wiki/福岡
- https://ja.wikipedia.org/wiki/名古屋
- https://ja.wikipedia.org/wiki/仙台
- https://ja.wikipedia.org/wiki/日光
- https://ja.wikipedia.org/wiki/箱根
- https://ja.wikipedia.org/wiki/伊勢神宮
- https://ja.wikipedia.org/wiki/清水寺

### standard-14
- https://ja.wikipedia.org/wiki/日本の世界遺産
- https://ja.wikipedia.org/wiki/法隆寺
- https://ja.wikipedia.org/wiki/姫路城
- https://ja.wikipedia.org/wiki/白川郷
- https://ja.wikipedia.org/wiki/厳島神社
- https://ja.wikipedia.org/wiki/原爆ドーム
- https://ja.wikipedia.org/wiki/知床
- https://ja.wikipedia.org/wiki/屋久島
- https://ja.wikipedia.org/wiki/富岡製糸場
- https://ja.wikipedia.org/wiki/富士山
- https://ja.wikipedia.org/wiki/古都京都の文化財
- https://ja.wikipedia.org/wiki/中尊寺

### advanced-14
- https://ja.wikipedia.org/wiki/紀伊山地の霊場と参詣道
- https://ja.wikipedia.org/wiki/石見銀山
- https://ja.wikipedia.org/wiki/平泉
- https://ja.wikipedia.org/wiki/明治日本の産業革命遺産
- https://ja.wikipedia.org/wiki/国立公園
- https://ja.wikipedia.org/wiki/日本の国立公園
- https://ja.wikipedia.org/wiki/上高地
- https://ja.wikipedia.org/wiki/尾瀬
- https://ja.wikipedia.org/wiki/阿蘇山
- https://ja.wikipedia.org/wiki/桜島
- https://ja.wikipedia.org/wiki/宗谷岬
- https://ja.wikipedia.org/wiki/波照間島

---

## スポーツ (sport)

### core-14
- https://ja.wikipedia.org/wiki/大相撲
- https://ja.wikipedia.org/wiki/柔道
- https://ja.wikipedia.org/wiki/剣道
- https://ja.wikipedia.org/wiki/日本プロ野球
- https://ja.wikipedia.org/wiki/王貞治
- https://ja.wikipedia.org/wiki/長嶋茂雄
- https://ja.wikipedia.org/wiki/イチロー
- https://ja.wikipedia.org/wiki/大谷翔平
- https://ja.wikipedia.org/wiki/横綱
- https://ja.wikipedia.org/wiki/両国国技館
- https://ja.wikipedia.org/wiki/嘉納治五郎
- https://ja.wikipedia.org/wiki/Jリーグ

### standard-14
- https://ja.wikipedia.org/wiki/読売ジャイアンツ
- https://ja.wikipedia.org/wiki/阪神タイガース
- https://ja.wikipedia.org/wiki/セントラル・リーグ
- https://ja.wikipedia.org/wiki/パシフィック・リーグ
- https://ja.wikipedia.org/wiki/日本シリーズ
- https://ja.wikipedia.org/wiki/選抜高等学校野球大会
- https://ja.wikipedia.org/wiki/全国高等学校野球選手権大会
- https://ja.wikipedia.org/wiki/箱根駅伝
- https://ja.wikipedia.org/wiki/全日本柔道選手権大会
- https://ja.wikipedia.org/wiki/オリンピック競技大会
- https://ja.wikipedia.org/wiki/1964年東京オリンピック
- https://ja.wikipedia.org/wiki/2020年東京オリンピック

### advanced-14
- https://ja.wikipedia.org/wiki/相撲部屋
- https://ja.wikipedia.org/wiki/幕内
- https://ja.wikipedia.org/wiki/大関
- https://ja.wikipedia.org/wiki/横綱審議委員会
- https://ja.wikipedia.org/wiki/日本相撲協会
- https://ja.wikipedia.org/wiki/講道館
- https://ja.wikipedia.org/wiki/日本武道館
- https://ja.wikipedia.org/wiki/全日本剣道連盟
- https://ja.wikipedia.org/wiki/流鏑馬
- https://ja.wikipedia.org/wiki/合気道
- https://ja.wikipedia.org/wiki/空手道
- https://ja.wikipedia.org/wiki/居合道

---

## 科学技術・産業 (sci)

### core-14
- https://ja.wikipedia.org/wiki/トヨタ自動車
- https://ja.wikipedia.org/wiki/ソニーグループ
- https://ja.wikipedia.org/wiki/本田技研工業
- https://ja.wikipedia.org/wiki/日産自動車
- https://ja.wikipedia.org/wiki/新幹線
- https://ja.wikipedia.org/wiki/山中伸弥
- https://ja.wikipedia.org/wiki/はやぶさ2
- https://ja.wikipedia.org/wiki/JAXA
- https://ja.wikipedia.org/wiki/富岳_(スーパーコンピュータ)
- https://ja.wikipedia.org/wiki/理化学研究所
- https://ja.wikipedia.org/wiki/小柴昌俊
- https://ja.wikipedia.org/wiki/日本銀行

### standard-14
- https://ja.wikipedia.org/wiki/湯川秀樹
- https://ja.wikipedia.org/wiki/朝永振一郎
- https://ja.wikipedia.org/wiki/江崎玲於奈
- https://ja.wikipedia.org/wiki/南部陽一郎
- https://ja.wikipedia.org/wiki/益川敏英
- https://ja.wikipedia.org/wiki/日本のノーベル賞受賞者
- https://ja.wikipedia.org/wiki/カミオカンデ
- https://ja.wikipedia.org/wiki/スーパーカミオカンデ
- https://ja.wikipedia.org/wiki/はやぶさ_(探査機)
- https://ja.wikipedia.org/wiki/イトカワ
- https://ja.wikipedia.org/wiki/かぐや_(探査機)
- https://ja.wikipedia.org/wiki/H-IIAロケット

### advanced-14
- https://ja.wikipedia.org/wiki/中村修二
- https://ja.wikipedia.org/wiki/赤崎勇
- https://ja.wikipedia.org/wiki/天野浩
- https://ja.wikipedia.org/wiki/大隅良典
- https://ja.wikipedia.org/wiki/本庶佑
- https://ja.wikipedia.org/wiki/吉野彰
- https://ja.wikipedia.org/wiki/真鍋淑郎
- https://ja.wikipedia.org/wiki/リチウムイオン二次電池
- https://ja.wikipedia.org/wiki/青色発光ダイオード
- https://ja.wikipedia.org/wiki/オートファジー
- https://ja.wikipedia.org/wiki/京_(スーパーコンピュータ)
- https://ja.wikipedia.org/wiki/物質・材料研究機構

---

## 宗教・信仰 (relig)

### core-14
- https://ja.wikipedia.org/wiki/神道
- https://ja.wikipedia.org/wiki/仏教
- https://ja.wikipedia.org/wiki/神社
- https://ja.wikipedia.org/wiki/寺院
- https://ja.wikipedia.org/wiki/鳥居
- https://ja.wikipedia.org/wiki/初詣
- https://ja.wikipedia.org/wiki/伊勢神宮
- https://ja.wikipedia.org/wiki/出雲大社
- https://ja.wikipedia.org/wiki/明治神宮
- https://ja.wikipedia.org/wiki/靖国神社
- https://ja.wikipedia.org/wiki/東大寺
- https://ja.wikipedia.org/wiki/法隆寺

### standard-14
- https://ja.wikipedia.org/wiki/浄土宗
- https://ja.wikipedia.org/wiki/浄土真宗
- https://ja.wikipedia.org/wiki/曹洞宗
- https://ja.wikipedia.org/wiki/臨済宗
- https://ja.wikipedia.org/wiki/日蓮宗
- https://ja.wikipedia.org/wiki/真言宗
- https://ja.wikipedia.org/wiki/天台宗
- https://ja.wikipedia.org/wiki/親鸞
- https://ja.wikipedia.org/wiki/法然
- https://ja.wikipedia.org/wiki/空海
- https://ja.wikipedia.org/wiki/最澄
- https://ja.wikipedia.org/wiki/日蓮

### advanced-14
- https://ja.wikipedia.org/wiki/古事記
- https://ja.wikipedia.org/wiki/日本書紀
- https://ja.wikipedia.org/wiki/天照大神
- https://ja.wikipedia.org/wiki/素戔嗚尊
- https://ja.wikipedia.org/wiki/大国主
- https://ja.wikipedia.org/wiki/八幡神
- https://ja.wikipedia.org/wiki/稲荷神
- https://ja.wikipedia.org/wiki/修験道
- https://ja.wikipedia.org/wiki/山伏
- https://ja.wikipedia.org/wiki/密教
- https://ja.wikipedia.org/wiki/禅
- https://ja.wikipedia.org/wiki/道元

---

## 言語 (lang)

### core-14
- https://ja.wikipedia.org/wiki/日本語
- https://ja.wikipedia.org/wiki/ひらがな
- https://ja.wikipedia.org/wiki/カタカナ
- https://ja.wikipedia.org/wiki/漢字
- https://ja.wikipedia.org/wiki/五十音
- https://ja.wikipedia.org/wiki/俳句
- https://ja.wikipedia.org/wiki/短歌
- https://ja.wikipedia.org/wiki/敬語
- https://ja.wikipedia.org/wiki/日本語の方言
- https://ja.wikipedia.org/wiki/仮名_(文字)
- https://ja.wikipedia.org/wiki/ローマ字
- https://ja.wikipedia.org/wiki/漢字の部首

### standard-14
- https://ja.wikipedia.org/wiki/常用漢字
- https://ja.wikipedia.org/wiki/教育漢字
- https://ja.wikipedia.org/wiki/尊敬語
- https://ja.wikipedia.org/wiki/謙譲語
- https://ja.wikipedia.org/wiki/丁寧語
- https://ja.wikipedia.org/wiki/古文
- https://ja.wikipedia.org/wiki/係り結び
- https://ja.wikipedia.org/wiki/枕詞
- https://ja.wikipedia.org/wiki/歌枕
- https://ja.wikipedia.org/wiki/万葉仮名
- https://ja.wikipedia.org/wiki/変体仮名
- https://ja.wikipedia.org/wiki/JIS_X_0208

### advanced-14
- https://ja.wikipedia.org/wiki/上代日本語
- https://ja.wikipedia.org/wiki/中古日本語
- https://ja.wikipedia.org/wiki/中世日本語
- https://ja.wikipedia.org/wiki/近世日本語
- https://ja.wikipedia.org/wiki/近代日本語
- https://ja.wikipedia.org/wiki/日本語の音韻
- https://ja.wikipedia.org/wiki/日本語の音節構造
- https://ja.wikipedia.org/wiki/沖縄方言
- https://ja.wikipedia.org/wiki/薩隅方言
- https://ja.wikipedia.org/wiki/津軽弁
- https://ja.wikipedia.org/wiki/京言葉
- https://ja.wikipedia.org/wiki/連濁
