#%%
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from keras.models import model_from_json
from keras.callbacks import TensorBoard
from os import path
from time import time
from model_pool import model_pool
import sys
import numpy as np
from math import ceil

model_ind = 1
folder_name = 'model_1'
if len(sys.argv) == 3:    
  try:
    model_ind = int(sys.argv[1])
    folder_name = f'{sys.argv[2]}_{model_ind}'
  except ValueError:
    model_ind = 1
    folder_name = 'model_1'

#folder_name = f'model_2nd_{model_ind}'
print('Folder name', folder_name)
config = {
        'epochs': 15,
        'trbatch_size': 128,
        'valbatch_size': 64,
        'tsbatch_size': 32,
        'input_shape': (150,150,3),
        'num_classes': 102
}
print(config)
print()

model = model_pool(model_ind, config)

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = path.join('data_split', 'train')
validation_dir = path.join('data_split', 'val')
test_dir = path.join('data_split', 'test')

train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=config['input_shape'][:-1],
        batch_size=config['trbatch_size'],
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=config['input_shape'][:-1],
        batch_size=config['valbatch_size'],
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=config['input_shape'][:-1],
        batch_size=config['tsbatch_size'],
        class_mode='categorical')

#%%
tensorboard = TensorBoard(log_dir=f'./{folder_name}', histogram_freq=0, write_graph=True, write_images=True)

start = time()
history = model.fit_generator(
      train_generator,
      steps_per_epoch=ceil(train_generator.samples // 
      config['trbatch_size']),
      epochs=config['epochs'],
      validation_data=validation_generator,
      validation_steps=ceil(validation_generator.samples // 
      config['valbatch_size']),
      callbacks=[tensorboard])
print()
print('Train duration:', time()-start)

start = time()
scores = model.evaluate_generator(
    generator=test_generator,
    steps=ceil(test_generator.samples // config['tsbatch_size'])
)
print('Test duration:', time()-start)

print("Accuracy: %.2f%%" % (scores[1]*100))
print('test loss:', scores[0])
print('test accuracy:', scores[1])

print(history.history.keys())
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

##Store Plots
#Accuracy plot
plt.plot(acc)
plt.plot(val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(f'{folder_name}/accuracy.png')
plt.close()

#Loss plot
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.savefig(f'{folder_name}/loss.png')

np.save(f'{folder_name}/acc.npy', acc)
np.save(f'{folder_name}/val_acc.npy', val_acc)
np.save(f'{folder_name}/loss.npy', loss)
np.save(f'{folder_name}/val_loss.npy', val_loss)


#Confusion Matrix
#Compute probabilities
Y_pred = model.predict_generator(test_generator,
  steps=(ceil(test_generator.samples // config['tsbatch_size'])))

#Assign most probable label
y_pred = np.argmax(Y_pred, axis=-1)
#Plot statistics
print()
print( 'Analysis of results' )
#target_names = train_generator.class_indices
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
print('class dict', [i for i in (test_generator.class_indices)])
print()
print('true classes', true_classes.shape, true_classes)
print()
print('y_pred', y_pred.shape, y_pred)
print()
print('true_classes-ypred', set(true_classes) - set(y_pred))
print()
label_map = (test_generator.class_indices)
label_map = dict((v,k) for k,v in label_map.items()) #flip k,v
print(label_map)
print()
predictions = [label_map[k] for k in y_pred]
print('predictions', predictions)
print()

print(classification_report(true_classes, y_pred,target_names=class_labels))
#print(confusion_matrix(true_classes, y_pred))

#Saving model and weights
model_json = model.to_json()
with open(f'{folder_name}/model.json', 'w') as json_file:
        json_file.write(model_json)
weights_file = f"{folder_name}/weights_"+str(scores[1])+".hdf5"
model.save_weights(weights_file, overwrite=True)

#Loading model and weights
json_file = open(f'{folder_name}/model.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights(weights_file)
