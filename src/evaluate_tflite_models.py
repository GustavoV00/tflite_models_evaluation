# src/04_evaluate_tflite_models.py
import numpy as np
import tensorflow as tf
import time
from pathlib import Path
import os
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

def evaluate_tflite_model(tflite_model_path, x_test, y_test):
    """Avalia um modelo TFLite no dataset de teste."""
    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
        
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        is_int8_input = input_details["dtype"] == np.int8 or input_details["dtype"] == np.uint8
        input_scale, input_zero_point = [0,0] 
        if is_int8_input and "quantization_parameters" in input_details:
            quant_params = input_details["quantization_parameters"]
            if quant_params and quant_params["scales"] and quant_params["zero_points"]: 
                 input_scale = quant_params["scales"][0]
                 input_zero_point = quant_params["zero_points"][0]


        predictions = []
        inference_times = []

        for i in range(len(x_test)):
            input_data_sample = x_test[i:i+1] 

            if is_int8_input and input_scale != 0: 
                input_data_sample = (input_data_sample / input_scale) + input_zero_point
                input_data_sample = input_data_sample.astype(input_details["dtype"])
            else:
                input_data_sample = input_data_sample.astype(np.float32)

            interpreter.set_tensor(input_details["index"], input_data_sample)

            start_time = time.perf_counter()
            interpreter.invoke()
            end_time = time.perf_counter()
            inference_times.append(end_time - start_time)

            output_data = interpreter.get_tensor(output_details["index"])
            
            predicted_label = np.argmax(output_data[0])
            predictions.append(predicted_label)

        avg_inference_time_ms = (sum(inference_times) / len(inference_times)) * 1000
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        precision_macro = report["macro avg"]["precision"]
        recall_macro = report["macro avg"]["recall"]
        f1_macro = report["macro avg"]["f1-score"]


        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_score_macro": f1_macro,
            "avg_inference_time_ms": avg_inference_time_ms,
            "classification_report": report 
        }
    except Exception as e:
        print(f"ERRO ao avaliar {tflite_model_path}: {e}")
        return None

if __name__ == "__main__":
    project_root = Path(__file__).parent
    tflite_models_load_dir = project_root / "models" / "tflite_models"
    data_load_path = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Carregando dados de teste pré-processados...")
    x_test = np.load(data_load_path / "x_test.npy")
    y_test = np.load(data_load_path / "y_test.npy")

    all_evaluation_results = []

    conversion_summary_path = results_dir / "tflite_conversion_summary.csv"
    df_conversions = None
    if conversion_summary_path.exists():
        df_conversions = pd.read_csv(conversion_summary_path)

    tflite_model_files = list(tflite_models_load_dir.glob("*.tflite"))
    if not tflite_model_files:
        print(f"Nenhum modelo TFLite encontrado em {tflite_models_load_dir}. Execute 03_convert_to_tflite.py primeiro.")
    else:
        for model_file_path in tflite_model_files:
            print(f"\nAvaliando TFLite Model: {model_file_path.name}...")
            metrics = evaluate_tflite_model(model_file_path, x_test, y_test)
            
            if metrics:
                model_base_name = model_file_path.name.replace(".tflite", "")
                parts = model_base_name.split("_")
                original_model_name = parts[0] + "_" + parts[1]
                quant_type = "_".join(parts[2:])

                result_entry = {
                    "tflite_model_name": model_file_path.name,
                    "original_model": original_model_name,
                    "quant_type": quant_type,
                    **metrics
                }
                
                # if df_conversions is not None:
                #     model_info_from_conversion = df_conversions[df_conversions["path"] == str(model_file_path)]
                #     if not model_info_from_conversion.empty:
                #         result_entry["size_kb"] = model_info_from_conversion.iloc[0]["size_kb"]
                #     else:
                #         result_entry["size_kb"] = os.path.getsize(model_file_path) / 1024.0


                all_evaluation_results.append(result_entry)
                print(f"  Acurácia: {metrics["accuracy"]:.4f}, Tempo Médio Inferência: {metrics["avg_inference_time_ms"]:.2f} ms")

    if all_evaluation_results:
        df_evaluations = pd.DataFrame(all_evaluation_results)
        print("\n--- Resumo da Avaliação dos Modelos TFLite ---")
        print(df_evaluations[["tflite_model_name", "accuracy", "avg_inference_time_ms", "precision_macro", "recall_macro", "f1_score_macro"]].round(4))
        
        eval_summary_path = results_dir / "tflite_evaluation_summary.csv"
        df_evaluations.to_csv(eval_summary_path, index=False)
        print(f"\nResumo da avaliação salvo em {eval_summary_path}")

        for res in all_evaluation_results:
            if "classification_report" in res:
                report_df = pd.DataFrame(res["classification_report"]).transpose()
                report_path = results_dir / f"class_report_{res["tflite_model_name"].replace(".tflite","")}.csv"
                report_df.to_csv(report_path)
                print(f"Relatório de classificação para {res["tflite_model_name"]} salvo em {report_path}")
    else:
        print("Nenhuma avaliação de TFLite realizada.")