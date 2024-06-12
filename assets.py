from typing import Literal


modelsT = Literal[
    "small", "medium", "large"
]


models_names: dict[modelsT, str] = {
    "small": "ru_core_news_sm",
    "medium": "ru_core_news_md",
    "large": "ru_core_news_lg",
}
base_dir_name = ".{model_type}_models"
models_dirs: dict[modelsT, str] = {
    "small": base_dir_name.format(model_type="small"),
    "medium": base_dir_name.format(model_type="medium"),
    "large": base_dir_name.format(model_type="large")
}
