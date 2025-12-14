# LLP_Compression

Для запуска сркиптов необходимо воспроизвести окружение с помощью команды: `pip install -r requirements.txt -q`

## Часть 1
### Сжатие модели

В данной работе используется post-training квантование модели, реализованное с помощью библиотеки llmcompressor. Квантованию подвергаются все линейные слои модели, кроме выходного слоя lm_head. Для квантования используется схема FP8 Dynamic, при которой веса представляются в 8-битном формате с динамическим масштабированием.

Чтобы сжать модель, необходимо запустить скрипт `compress.py`, в аргументах которого можно указать модель и директорию для сохранения полученных весов сжатой модели:

```
python compress.py \
  --model_id Qwen/Qwen3-8B \
  --output_dir Qwen3-8B-FP8-Dynamic
```

Веса сжатой модели:
- https://mega.nz/file/wMRTHCjJ#p-zdFx4pH1PMFBki7g7Ez9m5ceXWLpQDke_Jvli5Bow
- [Varya-K/Qwen3-8B-FP8-Dynamic](https://huggingface.co/Varya-K/Qwen3-8B-FP8-Dynamic)

### Оценка компрессии модели

Для оценки качества исходной и сжатой моделей используется бенчмарк MMLU. В рамках данной работы применяется стандартная реализация бенчмарка из библиотеки lm-eval. Метрика качества определяется как accuracy. Оценкой качества компресси определяется по следующей формуле:

$$ Performance \quad Drop = \frac{​Original \quad metric​−Compressed \quad metric}{Original \quad metric} $$

Так же происходит оценка сжатия модели, как отношение размера исходной модели к размеру сжатой:

$$ Compression \quad Ratio = \frac{Original \quad size}{Compressed \quad size}​ $$

Результирующая оценка компресии:

$$ Score = \frac{Compression \quad Ratio}{1 + Performance \quad Drop}​ $$

Для расчета описанных метрик необходимо запустить скрипт `evaluate_models.py`: `python evaluate_models.py`

Результаты оценки компресии:
