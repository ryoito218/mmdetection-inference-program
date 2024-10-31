import time
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
from mmengine.visualization import Visualizer
from mmengine.runner import Runner

image_path = "images/input/image_path" # 推論したい画像のパスを設定
model_weight_path = "work_dir/model_weight_path" # 重み付けモデルのパスを設定
config_path = "work_dir/config_path" # 設定ファイルのパスを設定

config = Config.fromfile(config_path) # 設定ファイルの読み込み
Runner.from_cfg(config) # ランナーのインスタンスの生成

image = mmcv.imread(image_path) # 画像の読み込み

model = init_detector(config, model_weight_path, device="cuda") # モデルの読み込み

start_time = time.time()
result = inference_detector(model, image) # 推論
end_time = time.time()

visualizer_now = Visualizer.get_current_instance() # 推論画像を生成するインスタンスを作成
visualizer_now.dataset_meta = model.dataset_meta # モデルのメタ情報を設定

# 推論画像の生成
visualizer_now.add_datasample(
    "result",
    image, # 推論する画像の設定
    data_sample=result, # 推論結果の設定
    draw_gt=False,
    wait_time=0,
    out_file="images/output/output.png", # 推論画像の出力先の設定
    pred_score_thr=0.3, # 検出の信頼度の設定
)

print(f"推論時間: {end_time-start_time}")