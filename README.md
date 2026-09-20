Определение ориентации текста

Решение для классификации текстовых кропов: обычная ориентация или поворот на 180°. Для каждого тестового изображения сохраняется вероятность p_180. Метрика соревнования — 1 − Brier, где Brier — среднее (p_180 − target)².

Подтверждённый результат на лидерборде: 0.96611242, Brier — 0.03388758. Он получен с дообученным PP-LCNet, объединением предсказаний двух ориентаций и температурной калибровкой.

Окружение

Команды выполняются из корня проекта. Нужны Linux, Python 3.12 и uv. Зависимости фиксируются в uv.lock; GPU-вариант проекта использует PaddlePaddle 3.3.1 из индекса CUDA 12.6.

uv sync --locked --extra gpu

Создайте .env по образцу .env.example, если файла ещё нет. Устройство можно задать строкой DEVICE=gpu:0; команды ниже передают его явно. Для CPU замените --extra gpu на --extra cpu, а --device gpu:0 — на --device cpu. Эти два набора зависимостей не устанавливаются вместе.

Тестовые данные должны лежать по следующим путям:

Содержимое

Путь

Образец ответа

data/test/sample_submission.csv

Изображения

data/test/test/images/test_00000.png … test_19999.png

Получение проверенного сабмишена

Для повторного инференса используются именно эти артефакты:

Артефакт

Путь

Дообученные веса

outputs/finetune_20260920_175555_827888/best.pdparams

Калибровка этих весов

outputs/calibration/finetuned_rotation.json

Отправленный CSV

outputs/submission_calibrated.csv

Настройки и хеши отправленного CSV

outputs/submission_calibrated.json

Каталоги outputs/ и weights/ исключены из Git. При переносе проекта веса и JSON калибровки нужно передать отдельно; скачивание исходных весов не заменяет дообученный checkpoint.

Команда ниже создаёт отдельный CSV, сохраняя уже отправленный файл:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.inference.predict \
  --weights outputs/finetune_20260920_175555_827888/best.pdparams \
  --calibration outputs/calibration/finetuned_rotation.json \
  --test-dir data/test \
  --batch-size 256 \
  --read-workers 8 \
  --device gpu:0 \
  --output outputs/submission_reproduced.csv

Результат — CSV с колонками image_id,p_180 и соседний JSON с настройками. Проверяются 20 000 строк, уникальность и состав ID, порядок из образца, конечность вероятностей и диапазон [0, 1]. Существующий результат можно заменить только с --overwrite.

SHA-256 артефактов подтверждённого запуска:

best.pdparams
ccf2a0cb194cc03b0801aab63fe46cd832bfa6641169ae3b330baabfc58062ba

submission_calibrated.csv
1b625c7ac0ecd657ac5fff09728c226921e05914f3d7dedb5fc97127aa4692e1

В записанном GPU-запуске чтение PNG, обработка и сохранение CSV заняли 17.357 с при batch size 256 и восьми потоках чтения. Загрузка модели и прогрев в этот замер не входят. Это измерение одного запуска, а не гарантия времени или побитовой идентичности на другом оборудовании.

Как получается вероятность

Изображение читается в RGB, приводится к размеру 160 × 80 и нормализуется средними [0.485, 0.456, 0.406] и стандартными отклонениями [0.229, 0.224, 0.225]. Модель работает в FP32. Класс 0 означает обычную ориентацию, класс 1 — поворот на 180°.

В режиме single используется вероятность класса 1 одного прохода. В режиме rotation модель получает исходное изображение и его поворот, выполненный до изменения размера:

q(x) = (p(x) + 1 - p(rot180(x))) / 2
p_180(x) = sigmoid(logit(q(x)) / T)

Температура применяется после объединения вероятностей. Для подтверждённого сабмишена T = 1.1390818601472776. Она подобрана по Brier только на реальных кропах calibration. JSON калибровки содержит хеш весов; инференс отклоняет файл, рассчитанный для другого checkpoint.

Данные для дообучения

Использованы HierText validation и синтетические строки. В текущем корпусе 66 271 реальный кроп и 4 000 синтетических: всего 70 271. Среди синтетических строк 3 000 кириллических и 1 000 латинских.

Подготовка с нуля или продолжение незавершённого запуска:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.data.prepare \
  --sources hiertext synthetic \
  --output data/external

Кропы и manifest.csv сохраняются в data/external/, загруженные исходники — в data/cache/. Шрифты DejaVu скачиваются автоматически в data/fonts/. Пути кропов в manifest относительны каталогу этого файла. --rebuild очищает результаты подготовки и заново создаёт корпус; для обычного продолжения он не нужен.

Проверка уже подготовленного корпуса без повторной генерации:

uv run --locked --extra gpu --env-file .env \
  python -m src.data.prepare --output data/external --check

По умолчанию проверка также открывает выборку из 64 PNG. Подготовка объединяет сцены с найденными дубликатами в группы и распределяет группы с seed 42 в пропорции 70/10/10/10. Поэтому доли по кропам могут отличаться от этих пропорций.

Часть

Исходные кропы

Примеры с двумя ориентациями

Назначение

train

48 743

97 486

Обновление весов

development

7 155

14 310

Выбор checkpoint и режима инференса

calibration

7 248

14 496

Подбор температуры для выбранных весов

holdout

7 125

14 250

Оценка зафиксированной конфигурации

PNG в корпусе уже приведены к обычной ориентации. Метки 0 и 1 создаются при обучении и оценке из пары «оригинал / поворот на 180°». Поле applied_rotation описывает исправление исходного кропа при подготовке и не является целевой меткой классификатора.

Для расширения предусмотрен источник rustitw: он подключается через --sources hiertext synthetic rustitw. Параметры доступны в python -m src.data.prepare --help; OCR для него находится в recognition.py. RusTitW не использовался в приведённых результатах.

Дообучение

Получение исходных специализированных весов, если их ещё нет:

uv run --locked --extra gpu --env-file .env \
  python -c "from src.models.weights import download_pretrained; print(download_pretrained())"

Для нового запуска выберите свободный --output. Команда проверенного запуска модуля обучения:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.training.train \
  --manifest data/external/manifest.csv \
  --weights weights/pretrained.pdparams \
  --output outputs/training \
  --epochs 6 \
  --batch-size 256 \
  --lr 0.001 \
  --seed 42 \
  --selection-variant rotation \
  --read-workers 8 \
  --device gpu:0

Обучаются все параметры модели. Используются cross-entropy, Momentum 0.9 с Nesterov, L2 4e-5 и ограничение глобальной нормы градиента до 5. LR разогревается примерно четверть эпохи, затем снижается по cosine до 0.0001. Каждая эпоха посещает обе ориентации каждого train-кропа. Случайные отражения и дополнительные произвольные повороты не применяются.

Train кэшируется в RAM как uint8; для текущего корпуса сам кэш занимает 3.49 GiB, сверх него нужна память процесса и рабочих массивов. Development читается при оценке. Кропы calibration и holdout обучение не использует.

Checkpoint выбирается по минимальному Brier на real-development в режиме rotation, без калибровки. Синтетика оценивается отдельно и не участвует в этом критерии.

В outputs/training/ сохраняются config.json, history.json, best.json, best.pdparams и каталоги завершённых эпох epoch_0001/ и далее. Каждый такой каталог содержит веса, состояние оптимизатора и метаданные. Продолжение прерванного запуска:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.training.train \
  --output outputs/training --resume --read-workers 8 --device gpu:0

Продолжение использует сохранённый план и последнюю полностью записанную эпоху. Незавершённая эпоха повторяется. Для другого числа эпох или LR нужен новый каталог; --resume не служит продлением завершённого плана. Побитовая воспроизводимость вычислений CUDA не гарантируется.

Оценка и калибровка новых весов

Ниже приведён порядок работы для нового checkpoint. Эти команды создают отдельные результаты и не меняют конфигурацию подтверждённого сабмишена. Для последующих экспериментов выбирайте новые пути результатов.

Сначала оценка на development:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.evaluation.evaluate \
  --weights outputs/training/best.pdparams \
  --manifest data/external/manifest.csv \
  --split development --device gpu:0 \
  --output outputs/evaluation/training_development

Без --calibration оцениваются single и rotation, отдельно для реальных и синтетических данных. Сохраняются predictions.csv.gz, metrics.csv и report.json с хешами входных файлов.

После выбора весов и режима инференса можно получить прогнозы на calibration и подобрать температуру:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.evaluation.evaluate \
  --weights outputs/training/best.pdparams \
  --manifest data/external/manifest.csv \
  --split calibration --device gpu:0 \
  --output outputs/evaluation/training_calibration

uv run --locked --extra gpu --env-file .env \
  python -m src.evaluation.calibrate \
  --evaluation outputs/evaluation/training_calibration \
  --variant rotation \
  --output outputs/calibration/training_rotation.json

Температура подбирается на реальных примерах calibration. Метрики до и после подбора на этой же части не являются независимой оценкой улучшения. Чужую температуру к новым весам применять нельзя.

Оценка зафиксированной конфигурации на holdout:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.evaluation.evaluate \
  --weights outputs/training/best.pdparams \
  --manifest data/external/manifest.csv \
  --calibration outputs/calibration/training_rotation.json \
  --split holdout --device gpu:0 \
  --output outputs/evaluation/training_holdout

Создание отдельного сабмишена с этими новыми весами:

uv run --locked --extra gpu --env-file .env \
  python -u -m src.inference.predict \
  --weights outputs/training/best.pdparams \
  --calibration outputs/calibration/training_rotation.json \
  --test-dir data/test --batch-size 256 --read-workers 8 --device gpu:0 \
  --output outputs/submission_training.csv

Зафиксированные результаты

Метрики checkpoint, использованного для подтверждённого сабмишена:

Выборка

Режим

Brier ↓

1 − Brier ↑

Development, real

single, T = 1

0.02472064

0.97527936

Development, real

rotation, T = 1

0.02077580

0.97922420

Calibration, real

rotation, T = 1

0.02258039

0.97741961

Calibration, real

rotation, подобранная T

0.02252003

0.97747997

Holdout, real

rotation, фиксированная T

0.02628072

0.97371928

Holdout, synthetic

rotation, фиксированная T

0.00339131

0.99660869

Лидерборд

rotation, фиксированная T

0.03388758

0.96611242

Предыдущий отправленный CSV получил 0.95800658. После перехода к объединению ориентаций и калибровке результат вырос на 0.00810584. Это совместное изменение; отдельный эффект температуры на лидерборде не измерялся.

Отдельный запуск src.training.train успешно завершил шесть эпох:

Эпоха

Train loss

Real-development Brier, rotation

1

0.229491

0.02794966

2

0.076044

0.02323639

3

0.054176

0.02265054

4

0.039427

0.02230648

5

0.031349

0.02217974

6

0.026398

0.02235824

Выбрана эпоха 5, веса — outputs/training/best.pdparams. Их SHA-256:

5b2c727396a0168316a397381758149e2fb969c6739ff57cdc0521c61372dbf5

Этот checkpoint уступает подтверждённому на real-development: 0.02217974 против 0.02077580. Его результат на лидерборде пока не измерен; балл 0.96611242 к нему не относится.

Подтверждённые веса получены более ранним отдельным запуском: тогда эпоха 4 была выбрана по cross-entropy на всём development. Текущий модуль выбирает по Brier на реальных данных, задаёт seed для каждой эпохи и выполняет поворот до resize. Поэтому команда нового обучения не воспроизводит прежнюю траекторию обновления весов. Для повторения подтверждённого инференса нужны сохранённые веса и калибровка из раздела выше.

Организация кода

Файлы

Назначение

src/data/prepare.py, sources.py, split.py

Общий запуск подготовки, загрузки, группы дубликатов и split

src/data/hiertext.py, synthetic.py, rustitw.py

Подготовка отдельных источников

src/data/recognition.py

OCR для проверки ориентации RusTitW

src/data/preprocessing.py

Чтение RGB и нормализация входа модели

src/models/lcnet.py, orientation.py, weights.py

Архитектура, загрузка модели, исходные веса

src/training/train.py

Дообучение, сохранение эпох, продолжение запуска

src/evaluation/metrics.py, evaluate.py, calibrate.py

Метрики, оценка двух ориентаций, подбор температуры

src/inference/predict.py, submission.py

Инференс и проверка итогового CSV

Ограничения оценки

В реальных данных преобладает латиница. Все 50 кропов с чистой кириллицей оказались в train; в остальных частях есть лишь единичные смешанные тексты. Высокое качество на кириллической синтетике не подтверждает такое же качество на реальных русскоязычных изображениях.

Группировка дубликатов по хешам уменьшает пересечения, но pHash не обнаруживает все кадрирования и сложные преобразования. Holdout уже оценивался; при дальнейшей настройке нельзя представлять его как ранее не просмотренную выборку. Модель и режим выбираются по development, температура — по calibration. Метрики внешнего корпуса и лидерборда описывают разные наборы изображений.