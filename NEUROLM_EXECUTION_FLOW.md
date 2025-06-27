# NeuroLM プロジェクト コード実行フロー詳細

このドキュメントでは、NeuroLM プロジェクトのコード実行順序を、前処理からインストラクションチューニング、推論まで、各ファイルの役割と連携を含めて詳細に説明します。

---

## 👑 全体像: NeuroLMの学習・推論パイプライン

NeuroLM プロジェクトは、大きく分けて以下の5つのフェーズで構成されます。

1. **データ準備**: 生のEEGデータやテキストデータを、モデルが学習できる形式に前処理します。
2. **VQモデル学習**: EEG信号を離散的な「EEGトークン」に変換するためのVQモデルを学習します。
3. **NeuroLMモデル事前学習**: 大量のEEGトークンとテキストトークンを使って、NeuroLMの基盤モデルを自己教師あり学習で事前学習します。
4. **インストラクションチューニング**: 事前学習済みNeuroLMモデルを、特定のタスク（指示）をこなせるように微調整します。
5. **推論**: 学習済みのNeuroLMモデルを使って、新しいEEGデータとテキストプロンプトからテキストを生成します。

---

## 👑 ステップ1: データ準備 (Data Preparation)

モデルに食わせるための「ご飯」を作るフェーズです。EEGデータとテキストデータの2種類を準備します。

### 1.1. EEGデータの前処理

* **目的**: 生のEEGデータ（例: `.edf`, `.cnt`ファイル）を読み込み、フィルタリング、リサンプリング、チャンネル選択などの前処理を行い、モデルが扱いやすい固定長のEEGセグメントと対応するラベルを`.pkl`ファイルとして保存します。
* **主要なスクリプト**: `dataset_maker/` ディレクトリ内の以下のファイル群。
* `dataset_maker/prepare_HMC.py`
* `dataset_maker/prepare_SEED.py`
* `dataset_maker/prepare_TUAB.py`
* `dataset_maker/prepare_TUEV.py`
* `dataset_maker/prepare_TUH_pretrain.py`
* `dataset_maker/prepare_TUSL.py`
* `dataset_maker/prepare_workload.py`

* **実行フロー**:
* これらのスクリプトのほとんど（`prepare_TUAB.py`を除く）は、**`if __name__ == "__main__":` ブロックを持たず、スクリプト全体が上から順に実行されます**。
* **共通パターン**:

1. **定数・設定の定義**: ファイルの先頭で、入力データパス (`rawDataPath`, `root`)、出力パス (`dump_folder`, `out_dir`)、ドロップするチャンネルリスト (`drop_channels`)、標準チャンネル順序 (`chOrder_standard`)、フィルタリング周波数などのパラメータが定義されます。
2. **ヘルパー関数の定義**:

* `readEDF(fileName)` (または `preprocessing(cntFilePath)`): `mne` ライブラリを使って生のEEGファイル（`.edf`や`.cnt`）を読み込み、フィルタリング、ノッチフィルタリング、リサンプリングを行います。不要なチャンネルのドロップやチャンネル順序の再編成もここで行われます。
* `BuildEvents(…)`: 読み込んだEEGデータとアノテーションファイル（例: `.txt`, `.rec`）から、固定長のEEGセグメントと対応するラベルを抽出します。
* `save_pickle(object, filename)`: 処理済みのEEGセグメントとラベルをPythonの `pickle` 形式で `.pkl` ファイルとして保存します。

3. **メイン処理**:

* `os.walk` や `Path.rglob` を使って、指定された入力ディレクトリから対象となるEEGファイル（例: `.edf`）を再帰的に検索し、ファイルリスト (`edf_files`, `group`) を作成します。
* ファイルリストを訓練/評価/テスト用に分割します。
* `load_up_objects(fileList, …, OutDir)` (または `multiprocessing.Pool` を使った並列処理): 各EEGファイルに対して `readEDF` や `BuildEvents` を呼び出し、処理されたEEGセグメントを `save_pickle` を使って指定された出力ディレクトリに `.pkl` ファイルとして保存します。
* **特記事項**: `dataset_maker/prepare_TUAB.py` と `dataset_maker/prepare_TUH_pretrain.py` は、`if __name__ == "__main__":` ブロック内で `multiprocessing.Pool` を使用し、`split_and_dump` (TUAB) や `process` (TUH_pretrain) 関数を並列実行することで、大量のファイルを効率的に処理します。

### 1.2. テキストデータの前処理

* **目的**: 大規模なテキストデータセット（OpenWebText）をダウンロードし、トークン化してバイナリファイルとして保存します。これは後のインストラクションチューニングや事前学習で利用されます。
* **主要なスクリプト**: `text_dataset_maker/prepare.py`

* **実行フロー**:
* このスクリプトは **`if __name__ == "__main__":` ブロックを持ちます**。
* `huggingface datasets` ライブラリの `load_dataset("openwebtext")` を呼び出して、OpenWebTextデータセットをダウンロードします。
* データセットを訓練 (`train`) と検証 (`val`) スプリットに分割します。
* `process(example)` 関数を定義し、`tiktoken.get_encoding("gpt2")` を使ってGPT-2のBPE (Byte Pair Encoding) でテキストをトークンIDに変換し、`eot_token` (End-Of-Textトークン) を追加します。
* `split_dataset.map(…)` を使って、データセット全体を並列でトークン化します。
* トークン化されたデータ (`tokenized`) を、`np.memmap` を利用して効率的に `train.bin` と `val.bin` というバイナリファイルとして保存します。これにより、メモリに収まらない大規模なデータも扱えるようになります。

---

## 👑 ステップ2: VQ (Vector Quantizer) モデルの学習

EEG信号を離散的な「EEGトークン」に変換するための、いわば「EEG専門の辞書」を作るフェーズです。

* **主要なスクリプト**: `train_vq.py`
* **モデル定義**: `model/model_vq.py` (`VQ_Align` クラス)
* **データセット**: `dataset.py` (`PickleLoader` クラス)

* **実行フロー**:

1. **`train_vq.py` の `main(args)` 関数が実行されます** (`if __name__ == "__main__":` ブロック内)。
2. **初期化**:

* `init(args)`: 分散学習 (DDP) のセットアップ、GPUデバイスの指定、乱数シードの設定など、学習環境を初期化します。
* チェックポイント保存用のディレクトリ (`checkpoints/VQ`) を作成します。
* `get_batch('train')`: `text_dataset_maker/prepare.py` で生成された `train.bin` からテキストデータを読み込むためのヘルパー関数。これはVQモデルのドメインアラインメント学習で利用されます。
* `PickleLoader`: ステップ1で準備されたEEGデータ（`.pkl`ファイル）を読み込むためのデータローダー (`dataset_train`) を初期化します。
* `VQ_Align` モデルのインスタンス化:
* `model/model_vq.py` の `VQ_Align` クラスが、`model/model_neural_transformer.py` の `NTConfig` を使ってエンコーダとデコーダの構成を定義し、インスタンス化されます。
* `VQ_Align` は内部で `model/norm_ema_quantizer.py` の `NormEMAVectorQuantizer` を使って量子化を行います。
* `torch.amp.GradScaler`: 混合精度学習のための `GradScaler` を初期化します。
* オプティマイザ (`AdamW`) を初期化します。
* 分散学習が有効な場合、モデルを `DDP` でラップします。
* WandBロギングをセットアップします。

3. **学習ループ**:

* `data_loader_train` からEEGデータバッチ (`X`, `Y_freq`, `Y_raw`, `input_chans`, `input_time`, `input_mask`) を取得します。
* `get_batch('train')` からテキストデータバッチ (`X_text`, `Y_text`) を取得します。
* **モデルのフォワードパス**:
* `model(X, Y_freq, Y_raw, input_chans, input_time, input_mask, alpha)`: EEGデータに対するVQモデルのフォワードパスを実行し、再構築損失、埋め込み損失、EEGドメイン分類損失 (`domain_loss`) を計算します。`alpha` は勾配反転層の強度を制御します。
* `model(X_text)`: テキストデータに対するドメイン分類損失 (`domain_loss2`) を計算します。
* これらの損失を合計し、`gradient_accumulation_steps` でスケールします。
* **バックワードパスと最適化**:
* `scaler.scale(loss).backward()`: 損失の勾配を計算します。
* `torch.nn.utils.clip_grad_norm_`: 勾配クリッピングを行います。
* `scaler.step(optimizer)`: オプティマイザをステップし、モデルのパラメータを更新します。
* `scaler.update()`: `GradScaler` を更新します。
* `optimizer.zero_grad()`: 勾配をクリアします。
* **ロギングとチェックポイント**: 定期的に訓練損失をログに出力し、モデルのチェックポイントを保存します。

4. **学習終了**: 分散学習プロセスを終了します。

---

## 👑 ステップ3: NeuroLMモデルの事前学習 (Pre-training)

VQモデルでEEGトークン化の準備ができたところで、いよいよNeuroLMモデル本体にEEGとテキストの「言語」を学ばせるフェーズです。

* **主要なスクリプト**: `train_pretrain.py`
* **モデル定義**: `model/model_neurolm.py` (`NeuroLM` クラス)
* **データセット**: `dataset.py` (`PickleLoader` クラス)

* **実行フロー**:

1. **`train_pretrain.py` の `main(args)` 関数が実行されます** (`if __name__ == "__main__":` ブロック内)。
2. **初期化**:

* `init(args)`: 分散学習 (DDP) のセットアップ、GPUデバイスの指定、乱数シードの設定など、学習環境を初期化します。
* チェックポイント保存用のディレクトリ (`checkpoints/NeuroLM-B`) を作成します。
* `get_batch('train')`: `text_dataset_maker/prepare.py` で生成された `train.bin` からテキストデータを読み込むためのヘルパー関数。
* `PickleLoader`: ステップ1で準備されたEEGデータ（`.pkl`ファイル）を読み込むためのデータローダー (`dataset_train`, `dataset_val`) を初期化します。
* **VQモデルのロードと固定**:
* `tokenizer_ckpt_path` から、ステップ2で学習済みの `VQ` モデル（`model/model_vq.py` の `VQ` クラス）をロードします。
* ロード後、`tokenizer.eval()` を呼び出し、`for p in self.tokenizer.parameters(): p.requires_grad = False` を実行することで、VQモデルのパラメータを固定し、学習中に更新されないようにします。
* `NeuroLM` モデルのインスタンス化:
* `model/model_neurolm.py` の `NeuroLM` クラスが、`model/model.py` の `GPTConfig` を使ってGPT部分の構成を定義し、インスタンス化されます。
* `init_from` 引数に応じて、スクラッチから初期化するか、OpenAIのGPT-2モデルから事前学習済み重みをロードします。
* `NeuroLM` は初期化時にロードされたVQモデルを `self.tokenizer` として内部に持ちます。
* `torch.amp.GradScaler` とオプティマイザ (`AdamW`) を初期化します。
* 分散学習が有効な場合、モデルを `DDP` でラップします。
* WandBロギングをセットアップします。

3. **学習ループ**:

* `data_loader_train` からEEGデータバッチ (`X_eeg`, `input_chans`, `input_time`, `input_mask`, `gpt_mask`, `num_chans`, `num_tokens`) を取得します。
* `get_batch('train')` からテキストデータバッチ (`X_text`, `Y_text`) を取得します。
* **EEGデータのトークン化**:
* `with torch.no_grad():` ブロック内で、**固定された** `tokenizer.get_codebook_indices(X_eeg, …)` を呼び出し、入力EEGデータから離散的な `codebook_indices` (EEGトークン) を抽出します。
* これらの `codebook_indices` を使って、`Y_eeg` (EEGトークンに対するターゲット) を構築します。
* **モデルのフォワードパス**:
* `model(X_eeg, Y_eeg, None, None, …)`: EEGデータとEEGトークンターゲットに対するフォワードパスを実行し、EEG損失 (`loss1`) を計算します。
* `model(None, None, X_text, Y_text)`: テキストデータとテキストトークンターゲットに対するフォワードパスを実行し、テキスト損失 (`loss2`) を計算します。
* `loss1` と `loss2` を合計し、`gradient_accumulation_steps` でスケールします。
* **バックワードパスと最適化**: ステップ2と同様に、勾配計算、クリッピング、オプティマイザステップ、勾配クリアを行います。
* **ロギングとチェックポイント**: 定期的に訓練損失をログに出力し、モデルのチェックポイントを保存します。

4. **評価**: 各エポックの終わりに、`evaluate` 関数を呼び出し、`data_loader_val` を使って検証セットでのモデルの性能（損失と精度）を評価します。
5. **学習終了**: 分散学習プロセスを終了します。

---

## 👑 ステップ4: インストラクションチューニング (Instruction Tuning)

事前学習で賢くなったNeuroLMモデルに、特定の「指示」に従う能力を教え込む、微調整（ファインチューニング）のフェーズです。

* **主要なスクリプト**: `train_instruction.py`
* **モデル定義**: `model/model_neurolm.py` (`NeuroLM` クラス)
* **データセット**: `downstream_dataset.py` (`SEEDDataset` など)、`utils.py` (`prepare_TUAB_dataset` など)

* **実行フロー**:

1. **`train_instruction.py` の `main(args)` 関数が実行されます** (`if __name__ == "__main__":` ブロック内)。
2. **初期化**:

* `init(args)`: 分散学習 (DDP) のセットアップ、GPUデバイスの指定、乱数シードの設定など、学習環境を初期化します。
* チェックポイント保存用のディレクトリ (`checkpoints/instruction-B`) を作成します。
* `get_batch('train')`: `text_dataset_maker/prepare.py` で生成された `train.bin` からテキストデータを読み込むためのヘルパー関数。
* **データセットの準備**:
* `get_instruct_datasets(args, name, …)` 関数を呼び出し、複数の下流タスクデータセット（SEED, TUAB, TUEV, TUSL, HMC, Workload）を準備します。
* これらの関数は、`downstream_dataset.py` の `SEEDDataset` や、`utils.py` の `prepare_TUAB_dataset` などを利用して、EEGデータとテキストデータ（指示と応答）のペアを読み込みます。
* `ConcatDataset` を使って、これらのデータセットを結合し、単一の訓練データローダー (`data_loader_merge`) を作成することも可能です。
* `NeuroLM` モデルのインスタンス化:
* `NeuroLM_path` から、ステップ3で学習済みの `NeuroLM` モデルの重みをロードします。
* `tokenizer_path` からVQモデルの重みをロードし、`NeuroLM` 内部のトークナイザとして設定します。
* `torch.amp.GradScaler` とオプティマイザ (`AdamW`) を初期化します。
* 分散学習が有効な場合、モデルを `DDP` でラップします。
* WandBロギングをセットアップします。

3. **学習ループ**:

* `data_loader_merge` (または個々のデータローダー) からバッチデータ (`X_eeg`, `X_text`, `Y_text`, `input_chans`, `input_time`, `input_mask`, `gpt_mask`) を取得します。
* `Y_eeg` は、この段階では通常、EEGトークンに対する直接的なターゲットとしては使われず、`-1` (無視) で埋められることが多いです。
* **モデルのフォワードパス**:
* `model(X_eeg, Y_eeg, X_text, Y_text, …)`: EEGとテキストのペアに対するフォワードパスを実行し、インストラクションチューニング損失 (`loss1`) を計算します。
* `model(None, None, X_text2, Y_text2)`: 別途取得した純粋なテキストデータに対するフォワードパスを実行し、テキスト損失 (`loss2`) を計算します。これは、モデルの言語モデルとしての能力を維持するためです。
* `loss1` と `loss2` を合計し、`gradient_accumulation_steps` でスケールします。
* **バックワードパスと最適化**: ステップ2、3と同様に、勾配計算、クリッピング、オプティマイザステップ、勾配クリアを行います。
* **ロギングとチェックポイント**: 定期的に訓練損失をログに出力し、モデルのチェックポイントを保存します。

4. **評価**: 各エポックの終わりに、`evaluate` 関数を呼び出し、各下流タスクの検証 (`data_loader_val`) およびテスト (`data_loader_test`) セットでモデルの性能を評価します。

* `evaluate` 関数は `model.generate(…)` を呼び出してテキストを生成します。
* 生成されたテキストは `tiktoken` でデコードされ、`get_pred` 関数でタスクに応じた予測（例: 分類ラベル）が抽出されます。
* `utils.py` の `get_metrics` 関数を使って、精度、F1スコアなどのメトリクスを計算します。

5. **学習終了**: 分散学習プロセスを終了します。

---

## 👑 ステップ5: 推論 (Inference)

学習済みのNeuroLMモデルを使って、実際に新しいEEGデータに対する予測や分析を行う最終フェーズです。

* **主要なスクリプト**: `simple_inference.py`
* **モデル定義**: `model/model_neurolm.py` (`NeuroLM` クラス)
* **データセット**: `downstream_dataset.py` (`SEEDDataset` など)

* **実行フロー**:

1. **`simple_inference.py` が直接実行されます** (`if __name__ == "__main__":` ブロック内)。
2. **`main(args)` 関数が呼び出されます**。
3. **引数のパース**: `argparse` を使って、コマンドライン引数（`--model_path`, `--tokenizer_path`, `--data_path`, `--output_dir`, `--sample_idx`）を読み込みます。
4. **デバイス設定**: `torch.cuda.is_available()` でGPUの有無を確認し、使用するデバイス (`cuda` または `cpu`) を設定します。
5. **出力ディレクトリ作成**: 推論結果を保存するための、タイムスタンプ付きのユニークな出力ディレクトリを作成します。
6. **モデルのロード**:

* `load_model(args.model_path, args.tokenizer_path, device)` 関数が呼び出されます。
* `model/model.py` の `GPTConfig` を使ってGPT部分の構成を定義します。
* `model/model_neurolm.py` の `NeuroLM` クラスをインスタンス化します。
* `args.model_path` から、ステップ4で学習済みの `NeuroLM` モデルのチェックポイントをロードし、モデルの `state_dict` に適用します。
* `args.tokenizer_path` からVQモデルのチェックポイントをロードし、`NeuroLM` 内部のトークナイザとして設定します。
* モデルを評価モード (`model.eval()`) に設定し、指定されたデバイスに移動します。

7. **データセットのロードとサンプル取得**:

* `downstream_dataset.py` の `SEEDDataset` をロードします。
* `args.sample_idx` で指定されたインデックスを使って、データセットから単一のEEGデータ (`eeg_data`)、テキストプロンプト (`text_prompt`)、対応するラベル (`label`)、およびその他の入力情報 (`input_chans`, `input_time`, `eeg_mask`, `gpt_mask`) を取得します。
* `dataset.get_ch_names()` でEEGチャンネル名を取得します。

8. **EEGデータの可視化**:

* `visualize_eeg(eeg_reshaped, channel_names, output_path)` 関数が呼び出され、入力EEG信号の波形がPNG画像として保存されます。

9. **推論の実行**:

* `run_inference(model, eeg_data, text_prompt, input_chans, input_time, eeg_mask, device)` 関数が呼び出されます。
* 入力データをモデルが期待する形式（バッチ次元の追加など）に整形し、デバイスに転送します。
* `model.generate(…)` メソッドを呼び出してテキストを生成します。このメソッドは `NeuroLM` クラスで定義されており、内部でEEGデータをVQトークンに変換し、それをテキストプロンプトと結合して `GPT` モデルに渡し、テキストを生成します。
* 生成にかかった時間を計測します。

10. **結果のデコードと表示**:

* `tiktoken.get_encoding("gpt2")` を使ってトークナイザを初期化します。
* `decode_tokens(tokens, tokenizer)` 関数を使って、生成されたトークン列を人間が読めるテキストにデコードします。
* 入力プロンプト、生成された応答、推論時間、真のラベルなどをコンソールに出力します。

11. **結果の保存**:

* 推論結果の詳細（タイムスタンプ、サンプルインデックス、入力プロンプト、生成された応答、推論時間、モデルパス、デバイス、EEGの形状など）をJSONファイルとして保存します。

12. **メモリ使用量の表示**: GPUが利用可能な場合、最大GPUメモリ使用量を表示します。
