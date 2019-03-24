from keras import layers
from keras import models
from keras import optimizers
from keras.constraints import maxnorm
from keras.applications import VGG19

def model_pool(model_ind, config):
  model = models.Sequential()

  # transfer learning
  if model_ind == 1:
    vgg = VGG19(weights='imagenet',
                include_top=False,
                input_shape=config['input_shape'])

    print(vgg.summary())
    vgg.trainable = True
    set_trainable = False
    for layer in vgg.layers:
      if layer.name == 'block5_conv1':
        set_trainable = True
      if set_trainable:
        layer.trainable = True
      else:
        layer.trainable = False

    model.add(vgg)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(102, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])

  elif model_ind == 2:
    # simple basic arch + bigger filters++ + remove some paddings + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])
    
  elif model_ind == 3:
    # simple basic arch + bigger filters++ + remove some paddings + layer , decay=0
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=0.0),
                  metrics=['acc'])

  elif model_ind == 4:
    # simple basic arch + bigger filters++ + remove some paddings + layer - SGD
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=config['lrate'], decay=config['lrate']//config['epochs'], momentum=0.7),
                  metrics=['acc'])
  
  elif model_ind == 5:
    # simple basic arch + bigger filters++ + remove some paddings + layer - Adam
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])

  elif model_ind == 9:
    # previous exp best model
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])
                  
  # best model dropouts
  '''
  if model_ind == 1:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 2:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Dropout(0.2, input_shape=config['input_shape']))
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 3:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Dropout(0.2, input_shape=config['input_shape']))
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])
  
  elif model_ind == 4:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 5:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu',  kernel_constraint=maxnorm(3)))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 6:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(3)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 7:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.Dropout(0.2))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu',  kernel_constraint=maxnorm(3)))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 8:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(3)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])
  
  elif model_ind == 9:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax',  kernel_constraint=maxnorm(4)))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=config['lrate'], decay=config['lrate']//config['epochs']),
                  metrics=['acc'])
  '''

  # variations of the best model (9)
  '''
  if model_ind == 1:
    # simple basic arch + bigger filters + no padding + layer
    model.add(layers.Conv2D(32, (7, 7), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (5, 5), activation='relu',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 2:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 3:
    # simple basic arch + equal big filters++ + padding + layer
    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 4:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (9, 9), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (9, 9), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (9, 9), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((9, 9)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 5:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(128, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 6:
    # simple basic arch + bigger filters++ + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])
  '''

  # consistent drop outs below to the best models
  '''
  if model_ind == 1:
    # cifar
    model.add(layers.Conv2D(32, (3, 3), input_shape=config['input_shape'], activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  if model_ind == 2:
    # (1)
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2))
    )
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 3:
    # (4) simple basic arch + bigger filters + maxpool
    model.add(layers.Conv2D(32, (7, 7), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 4:
    # (9) simple basic arch + bigger filters + padding + layer
    model.add(layers.Conv2D(32, (7, 7), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])
  '''

  # simple arch experiments below
  '''
  if model_ind == 1:
    # simple basic arch
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2))
    )
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 2:
    # simple basic arch - 1 layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2))
    )
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])


  elif model_ind == 3:
    # simple basic arch + layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 4:
    # simple basic arch + bigger filters + maxpool
    model.add(layers.Conv2D(32, (7, 7), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (7, 7), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 5:
    # simple basic arch + padding same
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 6:
    # custom
    # change simple basic arch + 3 layers + cifar arch
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 7:
    # change simple basic arch
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])


  elif model_ind == 8:
    # custom2 no padding + no double convs
    # change simple basic arch + 3 layers + cifar arch
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])


  elif model_ind == 9:
    # simple basic arch + bigger filters + padding + layer
    model.add(layers.Conv2D(32, (9, 9), activation='relu', padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (7, 7), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same',))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])

  elif model_ind == 10:
    # simple basic arch + more layers in fc and in conv
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                    input_shape=config['input_shape']))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(config['num_classes'], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=5e-4),
                  metrics=['acc'])
  '''

  # exploratory experiments below
  '''
  elif model_ind == 6:
    # cifar naive arch
    model.add(layers.Conv2D(32, (3, 3),
                    input_shape=config['input_shape']))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(config['num_classes']))
    model.add(layers.Activation('softmax'))

    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

  elif model_ind == 7:
    # cifar arch without dropouts
    model.add(layers.Conv2D(32, (3, 3),
                    input_shape=config['input_shape']))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(config['num_classes']))
    model.add(layers.Activation('softmax'))

    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])


  elif model_ind == 8:
    # cifar arch no dropouts bigger filter sizes
    model.add(layers.Conv2D(32, (7, 7),
                    input_shape=config['input_shape']))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (7, 7)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (5, 5)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (5, 5)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(config['num_classes']))
    model.add(layers.Activation('softmax'))

    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

  elif model_ind == 9:
    # cifar arch no dropouts extra pair of layers
    model.add(layers.Conv2D(32, (3, 3),
                    input_shape=config['input_shape']))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(config['num_classes']))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

  elif model_ind == 10:
    # cifar arch original
    model.add(layers.Conv2D(32, (3, 3), padding='same',
                    input_shape=config['input_shape']))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(config['num_classes']))
    model.add(layers.Activation('softmax'))

    opt = optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
  '''

  print('Model selected:', model_ind)
  print(model.summary())

  return model