# LLP_Compression

Для запуска сркиптов необходимо воспроизвести окружение с помощью команды: `pip install -r requirements.txt -q`

## Часть 1
### Сжатие модели

В данной работе применяется post-training квантование языковой модели с использованием библиотеки llmcompressor. Квантованию подвергаются все линейные слои модели, за исключением выходного слоя lm_head. В качестве метода квантования используется алгоритм AWQ (Activation-aware Weight Quantization) со схемой W4A16, при которой веса представляются в 4-битном целочисленном формате, а активации сохраняются в формате FP16. Квантование выполняется на основе калибровочного датасета без дополнительного дообучения модели.

Чтобы сжать модель, необходимо запустить скрипт `compress.py`, в аргументах которого можно указать модель, директорию для сохранения полученных весов сжатой модели, размер калибровочного датасета и максимальную длину последовательности для калибровки:

```
python compress.py \
  --model_id Qwen/Qwen3-8B \
  --output_dir Qwen3-8B-AWQ-INT4 \
  --num_calibration_samples 128 \
  --max_seq_length 1024
```

Веса сжатой модели:
- https://mega.nz/file/rF8zhajA#QC9DdOxpE2CpFgUIrj7q9Ei2Z2GQdogt6l-5vkgXqdc
- [Varya-K/Qwen3-8B-AWQ-INT4](https://huggingface.co/Varya-K/Qwen3-8B-AWQ-INT4)

### Оценка компрессии модели

Для оценки качества исходной и сжатой моделей используется бенчмарк MMLU. В рамках данной работы применяется стандартная реализация бенчмарка из библиотеки lm-eval. Метрика качества определяется как accuracy. Оценкой качества компресси определяется по следующей формуле:

$$ Performance \quad Drop = \frac{​Original \quad metric​−Compressed \quad metric}{Original \quad metric} $$

Так же происходит оценка сжатия модели, как отношение размера исходной модели к размеру сжатой:

$$ Compression \quad Ratio = \frac{Original \quad size}{Compressed \quad size}​ $$

Результирующая оценка компресии:

$$ Score = \frac{Compression \quad Ratio}{1 + Performance \quad Drop}​ $$

Для расчета описанных метрик необходимо запустить скрипт `evaluate_models.py`: `python evaluate_models.py`

Результаты оценки компресии:
```
==== RESULTS ====
Baseline accuracy: 0.754211
Compressed accuracy: 0.734737
Baseline size (MB): 15622.5881
Compressed size (MB): 5790.0920
Compression ratio: 2.6982
Performance drop: 0.025820
Score: 2.630286
```

## Часть 2
### Fine-tuning модели

Для fine-tuning модели была использована LoRA на датасете MMLU. Необходимо запустить скрипт `train.py`:
```
python train.py \
    --model_path "Varya-K/Qwen3-8B-AWQ-INT4" \
    --output_dir "Qwen3-8B-AWQ-INT4-TUNED" \
    --num_train_epochs 3 \
    --learning_rate 1e-3
```

Веса зафайнтюниной сжатой модели: [Varya-K/Qwen3-8B-AWQ-INT4-TUNED](https://huggingface.co/Varya-K/Qwen3-8B-AWQ-INT4-TUNED)

Результаты оценки:
```
==== RESULTS ====
Baseline accuracy: 0.754211
Tenude compressed accuracy: 0.746273
Baseline size (MB): 15622.5881
Tuned compressed size (MB): 5790.0920
Compression ratio: 2.6982
Performance drop: 0.010524
Score: 2.670058
```
