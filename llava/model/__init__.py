AVAILABLE_MODELS = {
    "f16": "F16ForCausalLM, F16Config",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
    except Exception as e:
        raise e
