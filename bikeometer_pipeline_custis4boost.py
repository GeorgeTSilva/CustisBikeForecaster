import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the class is labeled 'class' in the data file
tpot_data = np.recfromcsv('PATH/TO/DATA/FILE', delimiter='COLUMN_SEPARATOR', dtype=np.float64)
features = np.delete(tpot_data.view(np.float64).reshape(tpot_data.size, -1), tpot_data.dtype.names.index('class'), axis=1)
training_features, testing_features, training_classes, testing_classes = \
    train_test_split(features, tpot_data['class'], random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.85, learning_rate=0.1, loss="ls", max_depth=6, max_features=0.75, min_samples_leaf=6, min_samples_split=16, subsample=1.0)

exported_pipeline.fit(training_features, training_classes)
results = exported_pipeline.predict(testing_features)
