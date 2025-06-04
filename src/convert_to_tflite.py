import tensorflow as tf
import numpy as np
from pathlib import Path
import os

def representative_data_gen_mnist(x_train, num_samples=100):
    """Gerador de dados representativos para quantização INT8."""
    for i in range(num_samples):
        yield [x_train[i:i+1].astype(np.float32)]


def convert_and_save_tflite(keras_model_path, tflite_models_save_dir="tflite_models", x_train_for_int8_quant=None):
    keras_model_path = Path(keras_model_path)
    model_name_base = keras_model_path.stem
    
    tflite_save_dir = Path(tflite_models_save_dir)
    tflite_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Convertendo {model_name_base} para TFLite ---")
    model = tf.keras.models.load_model(keras_model_path)
    
    conversion_results = []

    try:
        converter_float32 = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model_float32 = converter_float32.convert()
        tflite_path_float32 = tflite_save_dir / f"{model_name_base}_float32.tflite"
        with open(tflite_path_float32, "wb") as f:
            f.write(tflite_model_float32)
        file_size = os.path.getsize(tflite_path_float32) / 1024
        conversion_results.append({"model": model_name_base, "type": "float32", "path": str(tflite_path_float32), "size_kb": file_size})
        print(f"  Float32: {tflite_path_float32} (Size: {file_size:.2f} KB)")
    except Exception as e:
        print(f"  ERRO na conversão Float32: {e}")

    # 3. Quantização Float16
    try:
        converter_float16 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_float16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_float16.target_spec.supported_types = [tf.float16]
        tflite_model_float16 = converter_float16.convert()
        tflite_path_float16 = tflite_save_dir / f"{model_name_base}_float16_quant.tflite"
        with open(tflite_path_float16, "wb") as f:
            f.write(tflite_model_float16)
        file_size = os.path.getsize(tflite_path_float16) / 1024
        conversion_results.append({"model": model_name_base, "type": "float16_quant", "path": str(tflite_path_float16), "size_kb": file_size})
        print(f"  Float16 Quant: {tflite_path_float16} (Size: {file_size:.2f} KB)")
    except Exception as e:
        print(f"  ERRO na quantização Float16: {e}")

    # 4. Quantização Inteira Completa (INT8)
    if x_train_for_int8_quant is not None:
        try:
            converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
            converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
            converter_int8.representative_dataset = lambda: representative_data_gen_mnist(x_train_for_int8_quant)
            converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter_int8.inference_input_type = tf.int8 # ou tf.uint8
            converter_int8.inference_output_type = tf.int8 # ou tf.uint8
            
            tflite_model_int8 = converter_int8.convert()
            tflite_path_int8 = tflite_save_dir / f"{model_name_base}_int8_quant.tflite"
            with open(tflite_path_int8, "wb") as f:
                f.write(tflite_model_int8)
            file_size = os.path.getsize(tflite_path_int8) / 1024
            conversion_results.append({"model": model_name_base, "type": "int8_quant", "path": str(tflite_path_int8), "size_kb": file_size})
            print(f"  INT8 Quant: {tflite_path_int8} (Size: {file_size:.2f} KB)")
        except Exception as e:
            print(f"  ERRO na quantização INT8: {e}")
    else:
        print("  INT8 Quant: Pulado (dados de calibração não fornecidos).")
        
    return conversion_results


if __name__ == "__main__":
    project_root = Path(__file__).parent
    print(project_root)
    keras_models_load_dir = project_root / "models" / "keras_models"
    tflite_models_save_dir = project_root / "models" / "tflite_models"
    data_load_path = project_root / "data"

    print("Carregando x_train para calibração INT8...")
    x_train_all = np.load(data_load_path / "x_train.npy")
    
    x_train_calibration = x_train_all[:1000] 

    all_conversion_stats = []

    keras_model_files = list(keras_models_load_dir.glob("*.h5"))
    for model_file in keras_model_files:
        stats = convert_and_save_tflite(str(model_file), 
                                        tflite_models_save_dir=tflite_models_save_dir, 
                                        x_train_for_int8_quant=x_train_calibration)
        all_conversion_stats.extend(stats)

    print("\n--- Resumo da Conversão TFLite ---")
    if all_conversion_stats:
        import pandas as pd
        df_conversions = pd.DataFrame(all_conversion_stats)
        print(df_conversions)
        results_dir = project_root / "src" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        df_conversions.to_csv(results_dir / "tflite_conversion_summary.csv", index=False)
        print(f"\nResumo da conversão salvo em {results_dir / "tflite_conversion_summary.csv"}")
    else:
        print("Nenhuma conversão realizada.")